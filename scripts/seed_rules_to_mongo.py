"""
ECGenius — scripts/seed_rules_to_mongo.py
==========================================
Seeds the MongoDB 'ontologyenginerules' collection (read by RuleExecutor,
see rules_engine/rule_executor.py) from ontology/rules_v2.csv.

Each CSV row is upserted by ruleId, so this script is safe to re-run after
editing rules_v2.csv.

Usage:
    python scripts/seed_rules_to_mongo.py
    python scripts/seed_rules_to_mongo.py --mongo-uri "mongodb+srv://..." --collection ontologyenginerules
    python scripts/seed_rules_to_mongo.py --csv ontology/rules.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def load_rule_rows(csv_path: Path) -> list[dict]:
    """Parse a rules CSV into OntologyEngineRule-shaped documents (camelCase)."""
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id or rule_id.startswith("#"):
                continue

            related_raw = row.get("related_labels", "") or ""
            symptom_raw = row.get("required_symptoms", "") or ""

            docs.append({
                "ruleId":           rule_id,
                "ruleType":         row.get("rule_type", "").strip().lower(),
                "primaryLabel":     row.get("primary_label", "").strip(),
                "relatedLabels":    [r.strip() for r in related_raw.split("|") if r.strip()],
                "requiredSymptoms": [s.strip() for s in symptom_raw.split("|") if s.strip()],
                "action":           row.get("action", "").strip(),
                "delta":            float(row.get("delta", 0) or 0),
                "version":          "v2",
                "active":           True,
            })
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Seed MongoDB ontologyenginerules collection from a rules CSV"
    )
    parser.add_argument("--mongo-uri", type=str, default=os.environ.get("MONGO_URI"),
                         help="MongoDB connection string (default: $MONGO_URI)")
    parser.add_argument("--db-name", type=str, default=os.environ.get("MONGO_DB_NAME"),
                         help="Database name (default: db embedded in --mongo-uri)")
    parser.add_argument("--collection", type=str, default="ontologyenginerules",
                         help="Target collection name (default: ontologyenginerules)")
    parser.add_argument("--csv", type=str, default="ontology/rules_v2.csv",
                         help="Rules CSV to seed from (default: ontology/rules_v2.csv)")
    args = parser.parse_args()

    if not args.mongo_uri:
        parser.error("No MongoDB URI provided — set MONGO_URI or pass --mongo-uri")

    csv_path = PROJECT_ROOT / args.csv
    if not csv_path.exists():
        parser.error(f"CSV not found: {csv_path}")

    docs = load_rule_rows(csv_path)
    if not docs:
        print(f"No rule rows found in {csv_path} — nothing to seed.")
        return

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name] if args.db_name else client.get_default_database()
    coll = db[args.collection]

    upserted = 0
    for doc in docs:
        coll.update_one({"ruleId": doc["ruleId"]}, {"$set": doc}, upsert=True)
        upserted += 1

    client.close()
    print(f"Seeded {upserted} rules from {csv_path} into "
          f"{db.name}.{args.collection}")


if __name__ == "__main__":
    main()
