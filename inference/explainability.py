"""
ECGenius — inference/explainability.py
=======================================
Generates human-readable, clinician-facing explanations for
every FusedResult from decision_fusion.py.

Two output levels:
  1. Short narrative  — one sentence per diagnosis for the UI card
  2. Full audit trail — structured breakdown for the detail panel

Input:  list[FusedResult]  from decision_fusion.py
Output: ExplainedResult list + UI payload dict + clinical report text

Design principles:
  - Never fabricate clinical facts
  - Tier-1 labels get explicit urgency prefix
  - Suppressed labels get a suppression reason
  - Score breakdown always shown verbatim
  - Every explanation is reproducible from FusedResult alone
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CONFIDENCE_PHRASES = {
    "CONFIRMED":  "strongly supported",
    "PROBABLE":   "likely",
    "POSSIBLE":   "possible but uncertain",
    "INCIDENTAL": "incidentally noted, low confidence",
}

TIER_URGENCY = {1: "CRITICAL", 2: "URGENT", 3: "ROUTINE"}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExplainedResult:
    label_id:             str
    label_name:           str
    confidence_label:     str
    score:                float
    tier:                 int
    short_narrative:      str
    score_breakdown_text: str
    audit_trail:          list[str]
    is_suppressed:        bool
    suppression_reason:   str
    default_action:       str
    aha_guideline:        str
    snomed_ct:            str
    icd10:                str
    supporting:           list[str] = field(default_factory=list)
    contradicting:        list[str] = field(default_factory=list)
    hierarchy:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label_id":             self.label_id,
            "label_name":           self.label_name,
            "confidence_label":     self.confidence_label,
            "score":                round(self.score, 3),
            "tier":                 self.tier,
            "urgency":              TIER_URGENCY.get(self.tier, ""),
            "short_narrative":      self.short_narrative,
            "score_breakdown_text": self.score_breakdown_text,
            "audit_trail":          self.audit_trail,
            "is_suppressed":        self.is_suppressed,
            "suppression_reason":   self.suppression_reason,
            "default_action":       self.default_action,
            "aha_guideline":        self.aha_guideline,
            "snomed_ct":            self.snomed_ct,
            "icd10":                self.icd10,
            "supporting":           self.supporting,
            "contradicting":        self.contradicting,
            "hierarchy_path":       " > ".join(self.hierarchy),
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Explainability:
    """
    Attach human-readable explanations to list[FusedResult].

    Parameters
    ----------
    institution_name : str
        Shown in audit trail and clinical report.
    include_suppressed : bool
        Whether to generate explanations for suppressed labels.
    """

    def __init__(
        self,
        institution_name:  str  = "ECGenius CDSS",
        include_suppressed: bool = True,
    ):
        self.institution       = institution_name
        self.include_suppressed = include_suppressed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_all(
        self,
        fused_results: list,          # list[FusedResult] from decision_fusion
        derived_log:   list[str] = None,
        patient_id:    str       = "unknown",
    ) -> list[ExplainedResult]:
        """
        Generate ExplainedResult for every FusedResult.

        Parameters
        ----------
        fused_results : list[FusedResult]
            Output of DecisionFusion.fuse()
        derived_log : list[str]
            From rule_executor.execute() — rule derivation log
        patient_id : str
            For audit trail header

        Returns
        -------
        list[ExplainedResult]
        """
        derived_log = derived_log or []
        explained   = []

        for fr in fused_results:
            if fr.is_suppressed and not self.include_suppressed:
                continue

            short      = self._short_narrative(fr, derived_log)
            breakdown  = self._score_breakdown(fr)
            audit      = self._audit_trail(fr, derived_log, patient_id)
            supp_reason = self._suppression_reason(fr, fused_results)

            explained.append(ExplainedResult(
                label_id=fr.label_id,
                label_name=fr.label_name,
                confidence_label=fr.confidence_label,
                score=fr.score,
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
                supporting=fr.supporting,
                contradicting=fr.contradicting,
                hierarchy=fr.hierarchy,
            ))

        logger.info("Explainability: generated %d explanations", len(explained))
        return explained

    def to_ui_payload(self, explained: list[ExplainedResult]) -> dict:
        """
        Serialise into the JSON payload consumed by ui/scripts/explain.js
        """
        active    = [e for e in explained if not e.is_suppressed]
        suppressed = [e for e in explained if e.is_suppressed]
        critical  = [e for e in active if e.tier == 1]

        return {
            "primary_diagnosis": active[0].to_dict() if active else None,
            "differential":      [e.to_dict() for e in active],
            "suppressed":        [e.to_dict() for e in suppressed],
            "critical_alerts":   [e.to_dict() for e in critical],
            "total_considered":  len(explained),
        }

    def to_json(self, explained: list[ExplainedResult], indent: int = 2) -> str:
        return json.dumps(self.to_ui_payload(explained), indent=indent)

    def format_clinical_report(
        self,
        explained:  list[ExplainedResult],
        patient_id: str = "unknown",
    ) -> str:
        """
        Plain-text clinical report suitable for EHR integration or PDF export.
        """
        active     = [e for e in explained if not e.is_suppressed]
        suppressed = [e for e in explained if e.is_suppressed]
        critical   = [e for e in active if e.tier == 1
                      and e.confidence_label in ("CONFIRMED", "PROBABLE")]

        lines = [
            "=" * 60,
            "ECGenius Clinical Decision Support Report",
            f"Institution: {self.institution}",
            f"Patient ID:  {patient_id}",
            "=" * 60,
            "",
        ]

        if critical:
            lines.append("!!! CRITICAL ALERTS !!!")
            for e in critical:
                lines.append(f"  [{e.confidence_label}] {e.label_name}")
                lines.append(f"  Action: {e.default_action}")
            lines.append("")

        lines.append("Differential Diagnosis (ranked):")
        lines.append("-" * 40)
        for i, e in enumerate(active, 1):
            lines.append(
                f"  {i}. {e.label_name:<35} "
                f"[{e.confidence_label}]  score={e.score:.3f}  Tier {e.tier}"
            )
            lines.append(f"     {e.short_narrative}")
            if e.supporting:
                lines.append(f"     Supporting: {', '.join(e.supporting)}")
            if e.contradicting:
                lines.append(f"     Against:    {', '.join(e.contradicting)}")
            if e.snomed_ct:
                lines.append(f"     SNOMED-CT: {e.snomed_ct}  ICD-10: {e.icd10}")
            lines.append("")

        if suppressed:
            lines.append("Considered but suppressed:")
            for e in suppressed:
                lines.append(f"  - {e.label_name}: {e.suppression_reason}")
            lines.append("")

        lines += [
            "=" * 60,
            "This report is AI-assisted and must be reviewed by a",
            "qualified clinician before clinical action is taken.",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    def _short_narrative(self, fr, derived_log: list[str]) -> str:
        if fr.is_suppressed:
            return (
                f"{fr.label_name} was considered but suppressed by an "
                f"ontology mutual-exclusion or precedence rule."
            )

        urgency           = TIER_URGENCY.get(fr.tier, "")
        confidence_phrase = CONFIDENCE_PHRASES.get(fr.confidence_label, fr.confidence_label)
        parts             = []

        if fr.tier == 1:
            parts.append(f"[{urgency}]")

        parts.append(
            f"{fr.label_name} is {confidence_phrase} "
            f"(score {fr.score:.2f})."
        )

        drivers = []
        if fr.pai >= 0.30:
            drivers.append(
                f"ECG pattern strongly detected (AI probability {fr.pai / 0.5:.0%})"
            )
        elif fr.pai >= 0.15:
            drivers.append(
                f"ECG pattern moderately detected ({fr.pai / 0.5:.0%})"
            )
        else:
            drivers.append(
                f"ECG pattern weakly detected ({fr.pai / 0.5:.0%})"
            )

        if fr.s_symptom > 0.05:
            drivers.append("supported by clinical symptoms")
        if fr.s_risk > 0.02:
            drivers.append("risk factor profile consistent")
        if fr.s_rule > 0.05:
            drivers.append("clinical rules boosted confidence")
        elif fr.s_rule < -0.05:
            drivers.append("clinical rules reduced confidence")

        # Check if derived
        if any(fr.label_id in e for e in derived_log):
            drivers.append("label boosted by derived clinical rule")

        parts.append(" — " + ", ".join(drivers) + ".")

        if fr.default_action:
            parts.append(f"Recommended action: {fr.default_action}.")

        return " ".join(parts)

    def _score_breakdown(self, fr) -> str:
        raw_pai = fr.pai / 0.5 if fr.pai > 0 else 0
        lines = [
            f"Score breakdown for {fr.label_id}:",
            f"  AI model  (0.5 × {raw_pai:.3f})  =  {fr.pai:.3f}",
            f"  Symptoms                        =  {fr.s_symptom:.3f}",
            f"  Risk factors                    =  {fr.s_risk:.3f}",
            f"  Rule / history deltas           =  {fr.s_rule:+.3f}",
            f"  {'─'*38}",
            f"  Final score                     =  {fr.score:.3f}  [{fr.confidence_label}]",
        ]
        return "\n".join(lines)

    def _audit_trail(self, fr, derived_log: list[str], patient_id: str) -> list[str]:
        raw_pai = fr.pai / 0.5 if fr.pai > 0 else 0
        trail = [
            f"Patient: {patient_id}",
            f"System:  {self.institution}",
            f"Label:   {fr.label_id} — {fr.label_name}",
            f"Hierarchy: {' > '.join(fr.hierarchy)}",
            f"Category:  {fr.category}",
            "",
            "── Evidence ──",
            f"  Raw Pai(D) from model: {raw_pai:.4f}",
            f"  S_ai      = 0.5 × {raw_pai:.4f} = {fr.pai:.4f}",
            f"  S_symptom = {fr.s_symptom:.4f}  (cap 0.30)",
            f"  S_risk    = {fr.s_risk:.4f}  (cap 0.10)",
            f"  S_rule    = {fr.s_rule:+.4f}  (rule + history deltas)",
            f"  Score     = {fr.score:.4f}  → {fr.confidence_label}",
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

        if fr.supporting:
            trail += ["", "── Supporting evidence ──"]
            trail += [f"  + {s}" for s in fr.supporting]

        if fr.contradicting:
            trail += ["", "── Contradicting evidence ──"]
            trail += [f"  - {c}" for c in fr.contradicting]

        relevant = [e for e in derived_log if fr.label_id in e]
        if relevant:
            trail += ["", "── Derived rule contributions ──"]
            trail += [f"  {e}" for e in relevant]

        if fr.evidence_log:
            trail += ["", "── History rule log ──"]
            trail += [f"  {e}" for e in fr.evidence_log]

        if fr.is_suppressed:
            trail += ["", f"── SUPPRESSED ──"]

        return trail

    def _suppression_reason(self, fr, all_results: list) -> str:
        if not fr.is_suppressed:
            return ""
        same_cat = [
            r for r in all_results
            if not r.is_suppressed and r.category == fr.category
            and r.label_id != fr.label_id
        ]
        if same_cat:
            winner = max(same_cat, key=lambda r: r.score)
            return (
                f"Suppressed by mutual-exclusion / precedence rule — "
                f"'{winner.label_id}' ({winner.label_name}) had higher "
                f"confidence in the same diagnostic group."
            )
        return "Suppressed by ontology rule."


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    from dataclasses import dataclass as dc

    @dc
    class StubFused:
        label_id: str; label_name: str; category: str; description: str
        hierarchy: list; pai: float; s_symptom: float; s_risk: float
        s_rule: float; score: float; confidence_label: str; tier: int
        default_action: str; allow_downgrade: bool; snomed_ct: str
        icd10: str; aha_guideline: str; clinical_notes: str
        is_suppressed: bool; is_derived: bool
        supporting: list; contradicting: list; evidence_log: list

    stubs = [
        StubFused("STEMI","ST-Elevation MI","Ischemia","Acute MI",
                  ["ROOT","Ischemia","STEMI"],0.37,0.28,0.05,0.10,0.80,
                  "CONFIRMED",1,"Immediate PCI referral",False,"57054005",
                  "I21","2013_AHA_STEMI","Time-critical",False,False,
                  ["chest pain","hypertension"],[],
                  ["[H1] chest pain → score +0.30"]),
        StubFused("NSR","Normal Sinus Rhythm","Rhythm","Normal",
                  ["ROOT","Rhythm","NSR"],0.30,0.0,0.0,0.0,0.30,
                  "POSSIBLE",2,"Routine review",True,"","","","",
                  True,False,[],[],[]),
    ]

    xai = Explainability(institution_name="AIIMS Delhi ECGenius",
                         include_suppressed=True)
    explained = xai.explain_all(stubs, ["R3: STEMI boosted"], patient_id="PT001")

    print("\n=== Short narratives ===\n")
    for e in explained:
        print(f"[{e.label_id}] {e.short_narrative}\n")

    print("\n=== Clinical report ===\n")
    print(xai.format_clinical_report(explained, patient_id="PT001"))