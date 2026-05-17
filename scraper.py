"""
SHL Catalog Auto-Update Scraper

Two modes:
  1. JSON refresh — Downloads the official SHL JSON catalog (fast, reliable).
  2. Full deep scrape — Crawls SHL website pages for enriched data (fact sheets, 
     sample reports, assessment length). This is the server-side equivalent of
     the browser console script.

Usage:
  python scraper.py                    # JSON refresh only (default)
  python scraper.py --full             # Full deep scrape + JSON refresh
  python scraper.py --schedule 24      # Auto-refresh every 24 hours
"""

import json
import csv
import time
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("⚠️ Install dependencies: pip install requests beautifulsoup4")
    print("   For JSON-only refresh, 'requests' alone is sufficient.")


# ── Configuration ───────────────────────────────────────────────────────────

CATALOG_JSON_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
SHL_CATALOG_BASE = "https://www.shl.com/products/product-catalog/"
OUTPUT_DIR = Path(__file__).parent
RATE_LIMIT_SECONDS = 0.5  # Delay between page requests


# ── JSON Refresh ────────────────────────────────────────────────────────────

def refresh_json_catalog(output_path: Optional[str] = None) -> List[dict]:
    """
    Download the latest official SHL catalog JSON.
    
    Returns:
        List of catalog entries.
    """
    if output_path is None:
        output_path = str(OUTPUT_DIR / "shl_product_catalog.json")

    print(f"📥 Downloading catalog from {CATALOG_JSON_URL}...")
    response = requests.get(CATALOG_JSON_URL, timeout=30)
    response.raise_for_status()

    catalog = response.json()
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(catalog)} items to {output_path}")
    return catalog


# ── Deep Scrape (Server-Side) ──────────────────────────────────────────────

def _scrape_product_page(url: str) -> dict:
    """Scrape detailed fields from an individual product page."""
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")

        def get_field_by_heading(heading_text: str) -> str:
            """Find an h4 heading and return the text of its next sibling."""
            headings = soup.select(".product-catalogue-training-calendar__row h4")
            for h in headings:
                if h.get_text(strip=True).lower() == heading_text.lower():
                    sibling = h.find_next_sibling()
                    if sibling:
                        return " ".join(sibling.get_text(strip=True).split())
            return "N/A"

        # Download links
        download_links = soup.select(".product-catalogue__downloads a")
        fact_sheet = "N/A"
        sample_report = "N/A"
        for a in download_links:
            text = a.get_text(strip=True).lower()
            href = a.get("href", "")
            if "fact sheet" in text and href:
                fact_sheet = href
            elif "sample report" in text and href:
                sample_report = href

        return {
            "Description": get_field_by_heading("Description"),
            "Job Levels": get_field_by_heading("Job levels"),
            "Languages": get_field_by_heading("Languages"),
            "Assessment Length": get_field_by_heading("Assessment length"),
            "Fact Sheet Link": fact_sheet,
            "Sample Report Link": sample_report,
        }
    except Exception as e:
        print(f"   ⚠️ Error scraping {url}: {e}")
        return {}


def _scrape_catalog_page(start: int = 0) -> List[dict]:
    """Scrape a single paginated catalog page."""
    url = f"{SHL_CATALOG_BASE}?type=1&start={start}"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr[data-entity-id]")

        if not rows:
            return []

        entries = []
        for row in rows:
            entity_id = row.get("data-entity-id", "")
            
            title_link = row.select_one(".custom__table-heading__title a")
            name = title_link.get_text(strip=True) if title_link else "N/A"
            detail_url = title_link.get("href", "") if title_link else ""
            if detail_url and not detail_url.startswith("http"):
                detail_url = f"https://www.shl.com{detail_url}"

            # Remote and Adaptive columns
            general_cols = row.select(".custom__table-heading__general")
            remote = "Yes" if general_cols and general_cols[0].select_one(".-yes") else "No"
            adaptive = "Yes" if len(general_cols) > 1 and general_cols[1].select_one(".-yes") else "No"

            # Test type keys
            test_types = [el.get_text(strip=True) for el in row.select(".product-catalogue__key")]

            entry = {
                "Entity ID": entity_id,
                "Individual Test Solutions": name,
                "Remote Testing": remote,
                "Adaptive/IRT": adaptive,
                "Test Type": ", ".join(test_types) if test_types else "None",
                "_detail_url": detail_url,
            }
            entries.append(entry)

        return entries
    except Exception as e:
        print(f"   ⚠️ Error scraping page at offset {start}: {e}")
        return []


def full_deep_scrape(output_path: Optional[str] = None) -> List[dict]:
    """
    Server-side deep scrape of the SHL catalog website.
    Equivalent to the browser console script but using requests + BeautifulSoup.
    
    Returns:
        List of fully enriched catalog entries.
    """
    if output_path is None:
        output_path = str(OUTPUT_DIR / "shl_catalog.csv")

    all_entries = []
    start = 0
    step = 12

    print("🔍 Starting deep scrape of SHL catalog website...")

    while True:
        print(f"   📄 Scraping page at offset {start}...")
        page_entries = _scrape_catalog_page(start)

        if not page_entries:
            break

        # Deep scrape each product page
        for i, entry in enumerate(page_entries):
            detail_url = entry.pop("_detail_url", "")
            if detail_url:
                print(f"      → Fetching details for '{entry['Individual Test Solutions']}'...")
                details = _scrape_product_page(detail_url)
                entry.update(details)
                time.sleep(RATE_LIMIT_SECONDS)

        all_entries.extend(page_entries)
        start += step
        time.sleep(RATE_LIMIT_SECONDS)

    # Write CSV
    if all_entries:
        headers = list(all_entries[0].keys())
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(all_entries)

        print(f"✅ Deep scrape complete: {len(all_entries)} items → {output_path}")
    else:
        print("⚠️ No items scraped.")

    return all_entries


# ── Scheduler ───────────────────────────────────────────────────────────────

def run_scheduled(interval_hours: float, full: bool = False):
    """Run the scraper on a schedule."""
    print(f"⏰ Scheduled mode: refreshing every {interval_hours} hours")
    print(f"   Mode: {'full deep scrape' if full else 'JSON refresh only'}")

    while True:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"\n{'='*60}")
            print(f"🔄 Catalog refresh at {timestamp}")
            print(f"{'='*60}")

            refresh_json_catalog()
            if full:
                full_deep_scrape()

            print(f"\n⏳ Next refresh in {interval_hours} hours...")
            time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped.")
            break
        except Exception as e:
            print(f"❌ Error during refresh: {e}")
            print("   Retrying in 5 minutes...")
            time.sleep(300)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHL Catalog Auto-Update Scraper"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full deep scrape (website crawl) in addition to JSON refresh",
    )
    parser.add_argument(
        "--schedule",
        type=float,
        default=0,
        help="Run on a schedule, refreshing every N hours (0 = run once)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only refresh the JSON catalog (default behavior)",
    )

    args = parser.parse_args()

    if args.schedule > 0:
        run_scheduled(args.schedule, full=args.full)
    else:
        refresh_json_catalog()
        if args.full:
            full_deep_scrape()
        print("\n🎉 Done!")


if __name__ == "__main__":
    main()
