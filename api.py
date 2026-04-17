"""
ECGenius — api.py
==================
FastAPI backend. Connects React frontend to the pipeline.

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations
import json
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ECGenius.API")

app = FastAPI(title="ECGenius API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline components once at startup ──────────────────────────────────
from inference.ontology_mapper        import OntologyMapper
from rules_engine.rule_executor       import RuleExecutor
from history_module.history_encoder   import HistoryEncoder
from inference.decision_fusion        import DecisionFusion

ONTOLOGY_DIR = str(PROJECT_ROOT / "ontology/")
HISTORY_DIR  = str(PROJECT_ROOT / "history_module/")
RULES_DIR    = str(PROJECT_ROOT / "rules_engine/")

mapper   = OntologyMapper(ontology_dir=ONTOLOGY_DIR)
executor = RuleExecutor(rules_dir=RULES_DIR, strict=False)
encoder  = HistoryEncoder(history_module_dir=HISTORY_DIR)
fusion   = DecisionFusion()


# ── Request / Response models ─────────────────────────────────────────────────

class PatientHistory(BaseModel):
    symptoms:     dict[str, bool]
    risk_factors: dict[str, bool]
    vitals:       dict[str, float]

class DiagnoseRequest(BaseModel):
    model_output: dict[str, float]   # {label_id: probability} from model.pt
    patient:      PatientHistory
    patient_id:   Optional[str] = "unknown"
    threshold:    Optional[float] = 0.10

class MockRequest(BaseModel):
    patient:    PatientHistory
    patient_id: Optional[str] = "PT_DEMO"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ECGenius API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema/{label_id}")
def get_schema(label_id: str):
    """Return what history questions to ask for a given label."""
    return encoder.questions_for_labels([label_id])

@app.get("/labels")
def get_labels():
    """Return all known leaf labels."""
    return {"labels": [
        {"id": lid, "name": meta.label_name, "category": meta.category}
        for lid, meta in mapper._labels.items()
        if meta.is_leaf
    ]}

@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    """Full pipeline: model output + patient history → ranked DDx."""
    try:
        patient = {
            "symptoms":     req.patient.symptoms,
            "risk_factors": req.patient.risk_factors,
            "vitals":       req.patient.vitals,
        }

        # Filter threshold
        model_output = {k: v for k, v in req.model_output.items()
                        if v >= req.threshold}

        # Pipeline
        results              = mapper.map(model_output)
        results, derived_log = executor.execute(results, patient, mapper)
        label_ids            = [r.label_id for r in results]
        history_deltas       = encoder.encode_all(label_ids, patient)
        output               = fusion.fuse(results, history_deltas,
                                           derived_log, patient, req.patient_id)

        return _serialize_output(output)

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose/mock")
def diagnose_mock(req: MockRequest):
    """Same pipeline but with hardcoded mock model probabilities."""
    mock_output = {
        "STEMI":        0.74,
        "AF":           0.42,
        "NSR":          0.61,
        "ST_Elevation": 0.80,
        "LVH":          0.38,
        "VF":           0.19,
    }
    full_req = DiagnoseRequest(
        model_output=mock_output,
        patient=req.patient,
        patient_id=req.patient_id,
    )
    return diagnose(full_req)


# ── Serialiser ────────────────────────────────────────────────────────────────

def _serialize_output(output) -> dict:
    def result_dict(r):
        return {
            "rank":             getattr(r, 'rank', 0),
            "label_id":         r.label_id,
            "label_name":       r.label_name,
            "category":         r.category,
            "hierarchy":        r.hierarchy,
            "score":            round(getattr(r, 'score_final', getattr(r, 'score', 0)), 3),
            "confidence_label": r.confidence_label,
            "tier":             r.tier,
            "tier_label":       getattr(r, 'tier_label', ''),
            "default_action":   r.default_action,
            "snomed_ct":        r.snomed_ct,
            "icd10":            r.icd10,
            "aha_guideline":    r.aha_guideline,
            "clinical_notes":   r.clinical_notes,
            "is_suppressed":    r.is_suppressed,
            "supporting":       getattr(r, 'supporting', []),
            "contradicting":    getattr(r, 'contradicting', []),
            "evidence_log":     getattr(r, 'evidence_log', []),
            "score_breakdown": {
                "s_ai":      round(getattr(r, 's_ai', getattr(r, 'pai', 0)), 3),
                "s_symptom": round(getattr(r, 's_symptom', 0), 3),
                "s_risk":    round(getattr(r, 's_risk', 0), 3),
                "s_rule":    round(getattr(r, 's_rule', 0), 3),
            },
        }

    active    = getattr(output, 'active_results', [r for r in output.results if not r.is_suppressed])
    suppressed = [r for r in output.results if r.is_suppressed]
    critical  = getattr(output, 'critical_alerts', [r for r in active if r.tier == 1])
    top       = getattr(output, 'top_diagnosis', active[0] if active else None)

    return {
        "patient_id":      getattr(output, 'patient_id', 'unknown'),
        "top_diagnosis":   result_dict(top) if top else None,
        "differential":    [result_dict(r) for r in active],
        "suppressed":      [result_dict(r) for r in suppressed],
        "critical_alerts": [result_dict(r) for r in critical],
        "derived_log":     getattr(output, 'derived_log', []),
        "total_considered": len(output.results),
        "metadata":        getattr(output, 'metadata', {}),
    }