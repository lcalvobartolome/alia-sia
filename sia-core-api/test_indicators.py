"""
Integration test for dashboard indicators (Q40, Q41, Q42).

Runs directly against the live Solr instance at http://kumo01:10085.
Execute from the project root:

    python test_dashboard_integration.py

Adjust SOLR_URL, CONFIG_FILE, DATE_START, DATE_END and FILTERS below
if needed.
"""

import json
import logging
import sys

# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

SOLR_URL    = "http://kumo01:10085"
CONFIG_FILE = "/export/usuarios_ml4ds/lbartolome/Repos/alia/alia-sia/sia-config/config.cf"

DATE_START  = "2025-01-01T00:00:00Z"
DATE_END    = "2026-01-01T00:00:00Z"
DATE_FIELD  = "updated"

# Try each of these tender_type values (None = all sources)
TENDER_TYPES = [None, "minors", "insiders", "outsiders"]

# A CPV prefix we know exists in the data
CPV_PREFIXES = ["31"]   # electrical equipment — matches 31680000 in the sample

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dashboard_test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _ok(label: str, result: dict) -> None:
    print(f"\n[OK] {label}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _fail(label: str, sc: int, result: dict) -> None:
    print(f"\n[FAIL] {label}  →  HTTP {sc}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _check(label: str, result, sc: int) -> bool:
    if sc != 200 or result is None or "error" in result:
        _fail(label, sc, result or {})
        return False
    _ok(label, result)
    return True


# ---------------------------------------------------------------------------
# Bootstrap the client
# ---------------------------------------------------------------------------

def _build_client():
    from src.core.clients.np_solr_client import SIASolrClient
    # Also import the three do_Q methods and bind them to the class so they
    # are available even if they have not been merged into the class yet.
    client =  SIASolrClient(logger=logger, config_file=CONFIG_FILE)
    client.solr_url = SOLR_URL
    print(client)
    return client

# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_q40(client, tender_type, cpv_prefixes=None):
    label = f"Q40 | tender_type={tender_type} | cpv={cpv_prefixes}"
    result, sc = client.do_Q40(
        date_start   = DATE_START,
        date_end     = DATE_END,
        date_field   = DATE_FIELD,
        tender_type  = tender_type,
        cpv_prefixes = cpv_prefixes,
    )
    ok = _check(label, result, sc)
    if ok:
        # Basic sanity checks
        assert "bimester_labels" in result, "missing bimester_labels"
        assert len(result["bimester_labels"]) == 6, (
            f"expected 6 bimesters, got {len(result['bimester_labels'])}"
        )
        assert result["total_tenders"] >= 0, "total_tenders must be >= 0"
        assert len(result["by_count"]) == len(result["bimester_labels"])
        assert len(result["by_budget"]) == len(result["bimester_labels"])
        print(f"    → total_tenders : {result['total_tenders']}")
        print(f"    → total_budget  : {result['total_budget']:.2f}")
    return ok


def test_q41(client, tender_type, cpv_prefixes=None):
    label = f"Q41 | tender_type={tender_type} | cpv={cpv_prefixes}"
    result, sc = client.do_Q41(
        date_start   = DATE_START,
        date_end     = DATE_END,
        date_field   = DATE_FIELD,
        tender_type  = tender_type,
        cpv_prefixes = cpv_prefixes,
    )
    ok = _check(label, result, sc)
    if ok:
        assert "bimester_labels" in result
        assert "pct_single_bid" in result
        assert "coverage" in result
        assert len(result["pct_single_bid"]) == len(result["bimester_labels"])
        non_null = [v for v in result["pct_single_bid"] if v is not None]
        if non_null:
            assert all(0 <= v <= 100 for v in non_null), (
                f"pct_single_bid out of range: {non_null}"
            )
            print(f"    → avg pct_single_bid : {sum(non_null)/len(non_null):.2f}%")
        else:
            print("    → no data (all None) — check offers_field name")
    return ok


def test_q42(client, tender_type, cpv_prefixes=None):
    label = f"Q42 | tender_type={tender_type} | cpv={cpv_prefixes}"
    result, sc = client.do_Q42(
        date_start   = DATE_START,
        date_end     = DATE_END,
        date_field   = DATE_FIELD,
        tender_type  = tender_type,
        cpv_prefixes = cpv_prefixes,
    )
    ok = _check(label, result, sc)
    if ok:
        assert "bimester_labels" in result
        assert "avg_days" in result
        assert "n_obs" in result
        non_null = [v for v in result["avg_days"] if v is not None]
        if non_null:
            assert all(v >= 0 for v in non_null), f"avg_days < 0: {non_null}"
            print(f"    → avg days overall : {sum(non_null)/len(non_null):.2f}")
        else:
            print("    → no data (all None) — check deadline/award field names")
    return ok


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_edge_cases(client):
    _section("Edge cases")

    # Empty result: CPV prefix that does not exist
    label = "Q40 | cpv=['99'] (should return zeros)"
    result, sc = client.do_Q40(
        date_start   = DATE_START,
        date_end     = DATE_END,
        cpv_prefixes = ["99"],
    )
    ok = _check(label, result, sc)
    if ok:
        assert result["total_tenders"] == 0, (
            f"expected 0 tenders for CPV 99, got {result['total_tenders']}"
        )

    # Budget range filter
    label = "Q40 | budget 100–500 EUR"
    result, sc = client.do_Q40(
        date_start = DATE_START,
        date_end   = DATE_END,
        budget_min = 100.0,
        budget_max = 500.0,
    )
    _check(label, result, sc)

    # Geography filter — value from the sample doc
    label = "Q40 | subentidad='Alicante/Alacant'"
    result, sc = client.do_Q40(
        date_start = DATE_START,
        date_end   = DATE_END,
        subentidad = "Alicante/Alacant",
    )
    _check(label, result, sc)

    # Contracting authority from the sample doc
    label = "Q40 | organo_id=L01030762"
    result, sc = client.do_Q40(
        date_start = DATE_START,
        date_end   = DATE_END,
        organo_id  = "L01030762",
    )
    _check(label, result, sc)

    # Narrow date range — single bimester
    label = "Q40 | date range = Ene–Feb 2025 only"
    result, sc = client.do_Q40(
        date_start = "2025-01-01T00:00:00Z",
        date_end   = "2025-03-01T00:00:00Z",
    )
    ok = _check(label, result, sc)
    if ok:
        assert len(result["bimester_labels"]) == 1, (
            f"expected 1 bimester, got {len(result['bimester_labels'])}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    passed = 0
    failed = 0

    logger.info("Building SIASolrClient...")
    try:
        client = _build_client()
    except Exception as e:
        logger.error(f"Could not instantiate SIASolrClient: {e}")
        sys.exit(1)

    # --- Q40 -----------------------------------------------------------------
    _section("Q40 – Total procurement")
    for tt in TENDER_TYPES:
        ok = test_q40(client, tender_type=tt)
        passed += ok; failed += not ok

    _section("Q40 – with CPV filter")
    ok = test_q40(client, tender_type=None, cpv_prefixes=CPV_PREFIXES)
    passed += ok; failed += not ok

    # --- Q41 -----------------------------------------------------------------
    _section("Q41 – Single bidder")
    for tt in TENDER_TYPES:
        ok = test_q41(client, tender_type=tt)
        passed += ok; failed += not ok

    _section("Q41 – with CPV filter")
    ok = test_q41(client, tender_type="minors", cpv_prefixes=CPV_PREFIXES)
    passed += ok; failed += not ok

    # --- Q42 -----------------------------------------------------------------
    _section("Q42 – Decision speed")
    for tt in TENDER_TYPES:
        ok = test_q42(client, tender_type=tt)
        passed += ok; failed += not ok

    # --- Edge cases ----------------------------------------------------------
    test_edge_cases(client)

    # --- Summary -------------------------------------------------------------
    _section("Summary")
    total = passed + failed
    print(f"\n  {passed}/{total} tests passed")
    if failed:
        print(f"  {failed} tests FAILED — review output above")
        sys.exit(1)
    else:
        print("  All tests passed ✓")


if __name__ == "__main__":
    main()