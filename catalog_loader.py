"""
Catalog Loader — Loads, merges, and normalizes the SHL product catalog.

Primary source: shl_product_catalog.json (official SHL-provided)
Enrichment source: shl_catalog.csv (scraped with additional detail fields)
"""

import json
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# ── Test type code mapping ──────────────────────────────────────────────────
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass
class CatalogItem:
    """Represents a single SHL assessment product."""
    entity_id: str
    name: str
    url: str  # Canonical catalog URL
    description: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration: str = ""
    remote: str = "yes"
    adaptive: str = "no"
    test_type_codes: str = ""       # Raw codes e.g. "K" or "A, P"
    test_type_labels: List[str] = field(default_factory=list)  # Full names
    fact_sheet_link: str = ""
    sample_report_link: str = ""
    # Pre-built search text for embedding
    search_text: str = ""

    def build_search_text(self) -> str:
        """Create a combined text blob optimized for semantic search."""
        parts = [
            self.name,
            self.description,
        ]
        if self.test_type_labels:
            parts.append("Test types: " + ", ".join(self.test_type_labels))
        if self.job_levels:
            parts.append("Job levels: " + ", ".join(self.job_levels))
        if self.duration:
            parts.append(f"Duration: {self.duration}")
        if self.remote == "yes":
            parts.append("Supports remote testing")
        if self.adaptive == "yes":
            parts.append("Adaptive/IRT enabled")
        if self.languages:
            # Include first few languages to keep text concise
            lang_str = ", ".join(self.languages[:5])
            if len(self.languages) > 5:
                lang_str += f" (+{len(self.languages) - 5} more)"
            parts.append(f"Languages: {lang_str}")

        self.search_text = " | ".join(parts)
        return self.search_text


def _parse_test_type_codes(keys_list: List[str]) -> tuple:
    """
    Convert the 'keys' field from JSON (full label names) to short codes.
    Returns (codes_string, labels_list).
    """
    # Reverse map: label -> code
    label_to_code = {v: k for k, v in TEST_TYPE_MAP.items()}
    # Also handle the exact keys from the JSON
    key_mapping = {
        "Ability & Aptitude": "A",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Assessment Exercises": "E",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S",
    }

    codes = []
    labels = []
    for key in keys_list:
        code = key_mapping.get(key, "")
        if code:
            codes.append(code)
            labels.append(key)
        else:
            # Try partial matching
            for label, c in key_mapping.items():
                if label.lower() in key.lower() or key.lower() in label.lower():
                    codes.append(c)
                    labels.append(label)
                    break

    return ", ".join(codes), labels


def _load_csv_enrichment(csv_path: str) -> Dict[str, dict]:
    """Load the scraped CSV and index by Entity ID for enrichment."""
    enrichment = {}
    if not os.path.exists(csv_path):
        return enrichment

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("Entity ID", "").strip()
            if eid:
                enrichment[eid] = {
                    "fact_sheet_link": row.get("Fact Sheet Link", "").strip(),
                    "sample_report_link": row.get("Sample Report Link", "").strip(),
                    "assessment_length": row.get("Assessment Length", "").strip(),
                    "csv_test_type": row.get("Test Type", "").strip(),
                }
    return enrichment


def load_catalog(
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> List[CatalogItem]:
    """
    Load and merge the SHL product catalog.
    
    Args:
        json_path: Path to shl_product_catalog.json (primary)
        csv_path: Path to shl_catalog.csv (enrichment)
    
    Returns:
        List of CatalogItem objects, ready for embedding.
    """
    base_dir = Path(__file__).parent

    if json_path is None:
        json_path = str(base_dir / "shl_product_catalog.json")
    if csv_path is None:
        csv_path = str(base_dir / "shl_catalog.csv")

    # 1. Load primary JSON catalog
    with open(json_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    # Use strict=False to handle embedded control characters in descriptions
    raw_catalog = json.loads(raw_text, strict=False)

    # 2. Load CSV enrichment
    csv_data = _load_csv_enrichment(csv_path)

    # 3. Build CatalogItem list
    items: List[CatalogItem] = []

    for entry in raw_catalog:
        eid = str(entry.get("entity_id", ""))
        keys = entry.get("keys", [])
        codes_str, labels = _parse_test_type_codes(keys)

        # Merge CSV enrichment if available
        csv_extra = csv_data.get(eid, {})

        # Determine duration — prefer JSON, fallback to CSV
        duration = entry.get("duration", "") or ""
        if not duration and csv_extra.get("assessment_length"):
            duration = csv_extra["assessment_length"]

        # If JSON has no test type codes, try CSV
        if not codes_str and csv_extra.get("csv_test_type"):
            codes_str = csv_extra["csv_test_type"]
            labels = [
                TEST_TYPE_MAP.get(c.strip(), c.strip())
                for c in codes_str.split(",")
                if c.strip()
            ]

        item = CatalogItem(
            entity_id=eid,
            name=entry.get("name", ""),
            url=entry.get("link", ""),
            description=entry.get("description", ""),
            job_levels=entry.get("job_levels", []),
            languages=entry.get("languages", []),
            duration=duration,
            remote=entry.get("remote", "yes"),
            adaptive=entry.get("adaptive", "no"),
            test_type_codes=codes_str,
            test_type_labels=labels,
            fact_sheet_link=csv_extra.get("fact_sheet_link", ""),
            sample_report_link=csv_extra.get("sample_report_link", ""),
        )
        item.build_search_text()
        items.append(item)

    print(f"✅ Catalog loaded: {len(items)} items")
    return items


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    catalog = load_catalog()
    for item in catalog[:3]:
        print(f"\n📦 {item.name} ({item.entity_id})")
        print(f"   URL: {item.url}")
        print(f"   Types: {item.test_type_codes} → {item.test_type_labels}")
        print(f"   Duration: {item.duration}")
        print(f"   Search: {item.search_text[:120]}...")
