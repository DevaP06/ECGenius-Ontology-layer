"""
ECGenius — inference/preprocess.py
=====================================
ECG signal preprocessing pipeline.

Steps (applied in order):
  1. Lead validation & reshaping  → (n_leads, n_samples)
  2. Bandpass filter              → remove baseline wander + HF noise
  3. Notch filter                 → 50/60 Hz powerline noise
  4. Resampling                   → target_fs (default 500 Hz)
  5. Segmentation                 → fixed-length window
  6. Normalisation                → per-lead z-score
  7. Quality check                → flag flat/noisy leads

Output: np.ndarray of shape (n_leads, segment_samples)
        ready to feed directly into the CNN-Transformer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not installed — bandpass/notch filters will be skipped. "
        "Run: pip install scipy"
    )


@dataclass
class PreprocessConfig:
    target_fs:          int   = 500     # Hz
    segment_length_sec: float = 10.0
    leads:              int   = 12
    bandpass_low:       float = 0.5     # Hz
    bandpass_high:      float = 40.0    # Hz
    notch_freq:         float = 50.0    # Hz (use 60.0 for US devices)
    notch_quality:      float = 30.0    # Q factor
    normalise:          bool  = True    # per-lead z-score


class Preprocessor:
    """
    Stateless ECG preprocessor. Instantiate once, call process() per patient.

    Parameters
    ----------
    target_fs, segment_length_sec, leads
        Forwarded to PreprocessConfig.
    source_fs : int | None
        Source sampling rate. If None, assumed equal to target_fs.
    """

    def __init__(
        self,
        target_fs: int = 500,
        segment_length_sec: float = 10.0,
        leads: int = 12,
        source_fs: int | None = None,
    ):
        self.cfg = PreprocessConfig(
            target_fs=target_fs,
            segment_length_sec=segment_length_sec,
            leads=leads,
        )
        self.source_fs = source_fs or target_fs
        self.segment_samples = int(target_fs * segment_length_sec)

    def process(self, ecg: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        ecg : np.ndarray
            Raw ECG. Accepted shapes:
              (n_leads, n_samples)   — standard multi-lead
              (n_samples, n_leads)   — transposed (auto-detected)
              (n_samples,)           — single-lead (replicated to n_leads)

        Returns
        -------
        np.ndarray
            Shape (n_leads, segment_samples), normalised, ready for model.
        """
        ecg = self._reshape(ecg)
        ecg = self._filter(ecg)
        ecg = self._resample(ecg)
        ecg = self._segment(ecg)
        ecg = self._normalise(ecg)
        self._quality_check(ecg)
        return ecg.astype(np.float32)

    # ------------------------------------------------------------------

    def _reshape(self, ecg: np.ndarray) -> np.ndarray:
        ecg = np.array(ecg, dtype=np.float64)

        if ecg.ndim == 1:
            logger.info("Single-lead input — replicating to %d leads.", self.cfg.leads)
            ecg = np.tile(ecg, (self.cfg.leads, 1))

        elif ecg.ndim == 2:
            # Detect transposed input: (n_samples, n_leads) → (n_leads, n_samples)
            if ecg.shape[0] > ecg.shape[1] and ecg.shape[1] == self.cfg.leads:
                ecg = ecg.T
            if ecg.shape[0] != self.cfg.leads:
                logger.warning(
                    "Expected %d leads, got %d — using first %d.",
                    self.cfg.leads, ecg.shape[0],
                    min(ecg.shape[0], self.cfg.leads),
                )
                ecg = ecg[: self.cfg.leads]
        else:
            raise ValueError(f"Unexpected ECG shape: {ecg.shape}")

        return ecg

    def _filter(self, ecg: np.ndarray) -> np.ndarray:
        if not SCIPY_AVAILABLE:
            return ecg

        fs = self.source_fs

        # Bandpass: remove baseline wander + high-freq noise
        low  = self.cfg.bandpass_low  / (0.5 * fs)
        high = self.cfg.bandpass_high / (0.5 * fs)
        low  = max(low, 1e-6)
        high = min(high, 0.9999)

        b, a = sp_signal.butter(4, [low, high], btype="band")
        ecg  = sp_signal.filtfilt(b, a, ecg, axis=1)

        # Notch: powerline interference
        b_n, a_n = sp_signal.iirnotch(
            self.cfg.notch_freq / fs, self.cfg.notch_quality
        )
        ecg = sp_signal.filtfilt(b_n, a_n, ecg, axis=1)

        return ecg

    def _resample(self, ecg: np.ndarray) -> np.ndarray:
        if self.source_fs == self.cfg.target_fs:
            return ecg
        if not SCIPY_AVAILABLE:
            logger.warning("scipy unavailable — skipping resampling.")
            return ecg

        target_samples = int(ecg.shape[1] * self.cfg.target_fs / self.source_fs)
        ecg = sp_signal.resample(ecg, target_samples, axis=1)
        logger.debug("Resampled %d → %d Hz (%d → %d samples)",
                     self.source_fs, self.cfg.target_fs,
                     ecg.shape[1], target_samples)
        return ecg

    def _segment(self, ecg: np.ndarray) -> np.ndarray:
        n = ecg.shape[1]
        target = self.segment_samples

        if n >= target:
            # Take the centre window (avoids lead-on artefacts)
            start = (n - target) // 2
            return ecg[:, start: start + target]
        else:
            # Pad with edge values (better than zero-padding for ECG)
            pad = target - n
            left  = pad // 2
            right = pad - left
            return np.pad(ecg, ((0, 0), (left, right)), mode="edge")

    def _normalise(self, ecg: np.ndarray) -> np.ndarray:
        if not self.cfg.normalise:
            return ecg

        mean = ecg.mean(axis=1, keepdims=True)
        std  = ecg.std(axis=1, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)   # avoid divide-by-zero on flat leads
        return (ecg - mean) / std

    def _quality_check(self, ecg: np.ndarray) -> None:
        flat_leads = []
        noisy_leads = []

        for i, lead in enumerate(ecg):
            amplitude = lead.max() - lead.min()
            if amplitude < 0.01:
                flat_leads.append(i)
            elif amplitude > 10.0:
                noisy_leads.append(i)

        if flat_leads:
            logger.warning(
                "Quality warning: flat signal detected on lead(s) %s. "
                "Check electrode placement.", flat_leads
            )
        if noisy_leads:
            logger.warning(
                "Quality warning: high amplitude on lead(s) %s — "
                "possible motion artefact.", noisy_leads
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(42)
    fake_ecg = rng.normal(0, 1, (12, 7500))   # 15s at 500Hz

    prep = Preprocessor(target_fs=500, segment_length_sec=10.0, leads=12)
    out  = prep.process(fake_ecg)

    print(f"\nInput shape:  {fake_ecg.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Mean:  {out.mean():.4f}  Std: {out.std():.4f}")
    print(f"dtype: {out.dtype}")