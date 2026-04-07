"""
ECGenius — inference/explainability.py
=======================================
Generates human-readable, clinician-facing explanations for
every FusionResult. This is the XAI layer.

Two levels of output:
  1. Short narrative  — one sentence per diagnosis for the UI card
  2. Full audit trail — structured breakdown for the detail panel

Design principles:
  - Never fabricate clinical facts — only describe what the
    scoring components actually contributed
  - Tier-1 labels get an explicit urgency prefix
  - Suppressed labels get a suppression reason
  - Score breakdown is always shown verbatim (no rounding secrets)
  - Every explanation is reproducible from the FusionResult alone
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence → plain English
CONFIDENCE_PHRASES = {
    "CONFIRMED":  "strongly supported",
    "PROBABLE":   "likely",
    "POSSIBLE":   "possible but uncertain",
    "INCIDENTAL": "incidentally noted, low confidence",
}

TIER_URGENCY = {
    1: "CRITICAL",
    2: "URGENT",
    3: "ROUTINE",
}


@dataclass
class ExplainedResult:
    """One result with short + full explanation attached."""
    label_id: str
    label_name: str
    confidence_label: str
    score_final: float
    tier: int
    short_narrative: str        # 1–2 sentences for UI card
    score_breakdown_text: str   # component-by-component plain text
    audit_trail: list[str]      # ordered list of reasoning steps
    is_suppressed: bool
    suppression_reason: str     # empty if not suppressed
    default_action: str
    aha_guideline: str
    snomed_ct: str
    icd10: str


class Explainability:
    """
    Attach human-readable explanations to FusionOutput.

    Parameters
    ----------
    institution_name : str
        Shown in audit trail header (e.g. "AIIMS Delhi")
    """

    def __init__(self, institution_name: str = "ECGenius CDSS"):
        self.institution = institution_name

    def explain(self, fusion_output) -> list[ExplainedResult]:
        """
        Parameters
        ----------
        fusion_output : FusionOutput
            From decision_fusion.py

        Returns
        -------
        list[ExplainedResult]
            Same order as fusion_output.results, with explanations attached.
            Also mutates fusion_output.results[n].explanation with the
            short narrative so the JSON output carries it.
        """
        explained = []

        for fr in fusion_output.results:
            short     = self._short_narrative(fr, fusion_output.derived_log)
            breakdown = self._score_breakdown(fr)
            audit     = self._audit_trail(fr, fusion_output)
            supp_reason = self._suppression_reason(fr, fusion_output.results)

            # Write back into FusionResult so to_json() includes it
            fr.explanation = short

            explained.append(ExplainedResult(
                label_id=fr.label_id,
                label_name=fr.label_name,
                confidence_label=fr.confidence_label,
                score_final=fr.score_final,
                tier=fr.tier,
                short_narrative=short,
                score_breakdown_text=breakdown,
                audit_trail=audit,
                is_suppressed=fr.is_suppressed,
                suppression_reason=supp_reason,
                default_action=fr.default_action,
                aha_guideline=fr.aha_guideline,
                snomed_ct=fr.snomed_ct,
                icd10=fr.icd10,
            ))

        logger.info("Explainability: generated explanations for %d labels", len(explained))
        return explained

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    def _short_narrative(self, fr, derived_log: list[str]) -> str:
        urgency   = TIER_URGENCY.get(fr.tier, "")
        confidence_phrase = CONFIDENCE_PHRASES.get(fr.confidence_label, fr.confidence_label)

        if fr.is_suppressed:
            return (
                f"{fr.label_name} was considered but suppressed by an ontology "
                f"mutual-exclusion or precedence rule."
            )

        parts = []

        # Urgency prefix for Tier-1
        if fr.tier == 1:
            parts.append(f"[{urgency}]")

        # Core sentence
        parts.append(
            f"{fr.label_name} is {confidence_phrase} "
            f"(score {fr.score_final:.2f})."
        )

        # What drove the score
        drivers = []
        if fr.s_ai >= 0.30:
            drivers.append(f"ECG pattern strongly detected (AI probability {fr.pai:.0%})")
        elif fr.s_ai >= 0.15:
            drivers.append(f"ECG pattern moderately detected ({fr.pai:.0%})")
        else:
            drivers.append(f"ECG pattern weakly detected ({fr.pai:.0%})")

        if fr.s_symptom > 0.05:
            drivers.append("supported by clinical symptoms")
        if fr.s_risk > 0.02:
            drivers.append("risk factor profile consistent")
        if fr.s_rule > 0.05:
            drivers.append("clinical rules boosted confidence")
        elif fr.s_rule < -0.05:
            drivers.append("clinical rules reduced confidence due to atypical presentation")

        parts.append(" — " + ", ".join(drivers) + ".")

        # Action
        if fr.default_action:
            parts.append(f"Recommended action: {fr.default_action}.")

        return " ".join(parts)

    def _score_breakdown(self, fr) -> str:
        lines = [
            f"Score breakdown for {fr.label_id}:",
            f"  AI model  (0.5 × {fr.pai:.3f})  =  {fr.s_ai:.3f}",
            f"  Symptoms                       =  {fr.s_symptom:.3f}",
            f"  Risk factors                   =  {fr.s_risk:.3f}",
            f"  Rule/history deltas            =  {fr.s_rule:+.3f}",
            f"  ──────────────────────────────────────",
            f"  Final score                    =  {fr.score_final:.3f}  [{fr.confidence_label}]",
        ]
        return "\n".join(lines)

    def _audit_trail(self, fr, fusion_output) -> list[str]:
        trail = [
            f"Patient: {fusion_output.patient_id}",
            f"System: {self.institution}",
            f"Label: {fr.label_id} — {fr.label_name}",
            f"Hierarchy: {' > '.join(fr.hierarchy)}",
            f"Category: {fr.category}",
            "",
            "── Evidence ──",
            f"  Pai(D) from CNN-Transformer: {fr.pai:.4f}",
            f"  S_ai  = 0.5 × {fr.pai:.4f} = {fr.s_ai:.4f}",
            f"  S_symptom = {fr.s_symptom:.4f}  (max 0.30)",
            f"  S_risk    = {fr.s_risk:.4f}  (max 0.10)",
            f"  S_rule    = {fr.s_rule:+.4f}  (rule/history deltas)",
            f"  Score     = {fr.score_final:.4f}  → {fr.confidence_label}",
            "",
            "── Triage ──",
            f"  Tier: {fr.tier} ({TIER_URGENCY.get(fr.tier, '?')})",
            f"  Default action: {fr.default_action}",
            f"  Allow downgrade: {fr.allow_downgrade}",
            "",
            "── Terminology ──",
            f"  SNOMED-CT: {fr.snomed_ct or 'N/A'}",
            f"  ICD-10:    {fr.icd10 or 'N/A'}",
            f"  Guideline: {fr.aha_guideline or 'N/A'}",
            f"  Notes:     {fr.clinical_notes or 'N/A'}",
        ]

        # Append relevant derived log entries
        relevant_derived = [e for e in fusion_output.derived_log if fr.label_id in e]
        if relevant_derived:
            trail += ["", "── Derived rule contributions ──"]
            trail += [f"  {e}" for e in relevant_derived]

        if fr.is_suppressed:
            trail += ["", f"── SUPPRESSED ── {self._suppression_reason(fr, fusion_output.results)}"]

        return trail

    def _suppression_reason(self, fr, all_results: list) -> str:
        if not fr.is_suppressed:
            return ""
        # Find the winner that caused suppression (same category, higher score)
        same_cat = [
            r for r in all_results
            if not r.is_suppressed
            and r.category == fr.category
            and r.label_id != fr.label_id
        ]
        if same_cat:
            winner = max(same_cat, key=lambda r: r.score_final)
            return (
                f"Suppressed by mutual-exclusion / precedence rule — "
                f"'{winner.label_id}' ({winner.label_name}) had higher confidence "
                f"in the same diagnostic group."
            )
        return "Suppressed by ontology rule (winner not identified in active results)."

    def format_clinical_report(self, fusion_output, explained: list[ExplainedResult]) -> str:
        """
        Generate a structured plain-text clinical report
        suitable for EHR integration or PDF export.
        """
        active = [e for e in explained if not e.is_suppressed]
        lines  = [
            f"{'='*60}",
            f"ECGenius Clinical Decision Support Report",
            f"Institution: {self.institution}",
            f"Patient ID:  {fusion_output.patient_id}",
            f"{'='*60}",
            "",
        ]

        # Critical alerts first
        critical = [e for e in active if e.tier == 1 and
                    e.confidence_label in ("CONFIRMED", "PROBABLE")]
        if critical:
            lines.append("!!! CRITICAL ALERTS !!!")
            for e in critical:
                lines.append(f"  [{e.confidence_label}] {e.label_name}")
                lines.append(f"  Action: {e.default_action}")
            lines.append("")

        # Ranked DDx
        lines.append("Differential Diagnosis (ranked):")
        lines.append("-" * 40)
        for i, e in enumerate(active, 1):
            lines.append(
                f"  {i}. {e.label_name:<35} "
                f"[{e.confidence_label}]  score={e.score_final:.2f}  "
                f"Tier {e.tier}"
            )
            lines.append(f"     {e.short_narrative}")
            lines.append("")

        # Suppressed
        suppressed = [e for e in explained if e.is_suppressed]
        if suppressed:
            lines.append("Considered but suppressed:")
            for e in suppressed:
                lines.append(f"  - {e.label_name}: {e.suppression_reason}")
            lines.append("")

        # Derived rule log
        if fusion_output.derived_log:
            lines.append("Rule engine log:")
            for entry in fusion_output.derived_log:
                lines.append(f"  {entry}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("This report is AI-assisted and must be reviewed by a")
        lines.append("qualified clinician before clinical action is taken.")
        lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    from dataclasses import dataclass as dc
    from typing import Optional as Opt

    @dc
    class StubFR:
        rank: int; label_id: str; label_name: str; category: str
        hierarchy: list; pai: float; s_ai: float; s_symptom: float
        s_risk: float; s_rule: float; score_final: float
        confidence_label: str; tier: int; tier_label: str
        default_action: str; allow_downgrade: bool
        snomed_ct: str = ""; icd10: str = ""; aha_guideline: str = ""
        clinical_notes: str = ""; is_suppressed: bool = False
        derived: bool = False; explanation: str = ""

    @dc
    class StubOutput:
        patient_id: str; results: list; active_results: list
        top_diagnosis: Opt[object]; critical_alerts: list; derived_log: list

    results = [
        StubFR(1,"STEMI","ST-Elevation MI","Ischemia",
               ["ROOT","Ischemia","STEMI"],0.55,0.275,0.20,0.05,0.30,0.825,
               "CONFIRMED",1,"life-threatening","Immediate PCI",False,
               snomed_ct="57054005",icd10="I21",
               aha_guideline="2013_AHA_STEMI_Guideline",
               clinical_notes="Time-critical diagnosis"),
        StubFR(2,"VF","Ventricular Fibrillation","Rhythm",
               ["ROOT","Rhythm","VF"],0.78,0.39,0.05,0.02,0.0,0.46,
               "POSSIBLE",1,"life-threatening","Start ACLS",False),
        StubFR(0,"AF","Atrial Fibrillation","Rhythm",
               ["ROOT","Rhythm","AF"],0.71,0.355,0.15,0.02,0.0,0.525,
               "PROBABLE",2,"urgent","Rate control",True,
               is_suppressed=True),
    ]

    output = StubOutput(
        patient_id="PT001",
        results=results,
        active_results=[r for r in results if not r.is_suppressed],
        top_diagnosis=results[0],
        critical_alerts=[results[0]],
        derived_log=["R3: STEMI boosted +0.30 (ST_Elevation + chest_pain)"],
    )

    xai = Explainability(institution_name="AIIMS Delhi ECGenius")
    explained = xai.explain(output)

    print("\n=== Short narratives ===\n")
    for e in explained:
        print(f"[{e.label_id}]")
        print(f"  {e.short_narrative}\n")

    print("\n=== Clinical report ===\n")
    print(xai.format_clinical_report(output, explained))