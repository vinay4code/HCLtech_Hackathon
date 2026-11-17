import sqlite3
import json
from pathlib import Path
from typing import Optional
from config.settings import DB_PATH, BASE_DIR

SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(force: bool = False):
    """Initializes the database schema and populates seed data if empty or forced."""
    if force and DB_PATH.exists():
        DB_PATH.unlink()

    conn = get_db_connection()
    # Check if uploaded_documents exists and lacks checksum_sha256
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(uploaded_documents)")
        cols = [r["name"] for r in cursor.fetchall()]
        if cols and "checksum_sha256" not in cols:
            cursor.execute("DROP TABLE uploaded_documents")
            conn.commit()

        # Check if approvals lacks confidence_score
        cursor.execute("PRAGMA table_info(approvals)")
        appr_cols = [r["name"] for r in cursor.fetchall()]
        if appr_cols and "confidence_score" not in appr_cols:
            cursor.execute("ALTER TABLE approvals ADD COLUMN confidence_score REAL DEFAULT 0.0")
            conn.commit()
    except Exception:
        pass

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()

    # Seed initial offers if offers table is empty
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM offers")
    if cursor.fetchone()[0] == 0:
        _seed_initial_offers(conn)
        _seed_initial_benchmarks(conn)
    _seed_initial_documents(conn)

    conn.close()

def _seed_initial_documents(conn: sqlite3.Connection):
    import hashlib
    from datetime import datetime
    from config.settings import OFFERS_DOCS_DIR
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM uploaded_documents")
    if cursor.fetchone()[0] == 0 and OFFERS_DOCS_DIR.exists():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for fpath in OFFERS_DOCS_DIR.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in [".pdf", ".docx", ".txt", ".csv"]:
                try:
                    with open(fpath, "rb") as f:
                        content = f.read()
                    chk = hashlib.sha256(content).hexdigest()
                    cursor.execute("""
                        INSERT OR IGNORE INTO uploaded_documents (
                            doc_id, filename, file_type, upload_timestamp, num_chunks, status, file_size_bytes, checksum_sha256
                        ) VALUES (?, ?, ?, ?, ?, 'INDEXED', ?, ?)
                    """, (fpath.name, fpath.name, fpath.suffix.upper().replace(".", ""), now, 3, len(content), chk))
                except Exception:
                    pass
        conn.commit()

def _seed_initial_offers(conn: sqlite3.Connection):
    seed_offers = [
        (
            "OFFER-LOYALTY-CREDIT",
            "Long-Term Loyalty Bill Credit",
            "LOYALTY",
            "MONTHLY_CREDIT",
            15.0,
            48,
            999,
            json.dumps(["Month-to-month", "One year", "Two year"]),
            json.dumps(["DSL", "Fiber optic", "No"]),
            json.dumps([]),
            "LONG_TERM_LOYAL",
            "2026-01-01",
            "2026-12-31",
            "Applicable to accounts active for 48+ months. $15 credit applied over 3 consecutive billing cycles. Non-transferable.",
            "loyalty_rewards_handbook.txt",
            1,
            1,
            "2026-01-01 00:00:00"
        ),
        (
            "OFFER-FIBER-BOOST-20",
            "Fiber Speed Boost 500Mbps",
            "UPGRADE",
            "PERCENTAGE",
            20.0,
            6,
            999,
            json.dumps(["One year", "Two year", "Month-to-month"]),
            json.dumps(["Fiber optic", "DSL"]),
            json.dumps([]),
            "UPSELL_OPPORTUNITY",
            "2026-01-01",
            "2026-12-31",
            "Requires compatible fiber gateway. Promotional 20% discount valid for 6 months. Standard rates apply thereafter.",
            "fiber_upgrade_promos.txt",
            1,
            1,
            "2026-01-01 00:00:00"
        ),
        (
            "OFFER-SECURITY-BUNDLE-10",
            "Complete Cyber Shield Bundle",
            "SECURITY",
            "FIXED_AMOUNT",
            10.0,
            3,
            999,
            json.dumps(["Month-to-month", "One year", "Two year"]),
            json.dumps(["Fiber optic", "DSL"]),
            json.dumps(["OnlineSecurity"]),
            "UPSELL_OPPORTUNITY",
            "2026-01-01",
            "2026-12-31",
            "Includes 24/7 antivirus protection and 250GB cloud backup. $10 discount applied each month while both services remain active.",
            "security_bundle_campaigns.csv",
            1,
            1,
            "2026-01-01 00:00:00"
        ),
        (
            "OFFER-ANNUAL-LOCKIN-15",
            "Annual Commitment Rate Lock",
            "RENEWAL",
            "PERCENTAGE",
            15.0,
            12,
            999,
            json.dumps(["Month-to-month"]),
            json.dumps(["DSL", "Fiber optic", "No"]),
            json.dumps([]),
            "CONTRACT_RENEWAL_CANDIDATE",
            "2026-01-01",
            "2026-12-31",
            "Requires transition from Month-to-month to minimum 12-month commitment. 15% discount guaranteed for 12 months.",
            "annual_renewal_discounts.txt",
            1,
            1,
            "2026-01-01 00:00:00"
        ),
        (
            "OFFER-WELCOME-STREAM-25",
            "New Subscriber StreamPass 25%",
            "ONBOARDING",
            "PERCENTAGE",
            25.0,
            0,
            6,
            json.dumps(["Month-to-month", "One year", "Two year"]),
            json.dumps(["Fiber optic", "DSL"]),
            json.dumps(["StreamingTV"]),
            "EARLY_STAGE_CUSTOMER",
            "2026-01-01",
            "2026-12-31",
            "Exclusive for subscribers in their first 6 months. 25% off StreamingTV or StreamingMovies for 3 consecutive months.",
            "sample_offers.pdf",
            2,
            1,
            "2026-01-01 00:00:00"
        ),
        (
            "OFFER-TECH-SUPPORT-PLUS",
            "24/7 Premium Tech Care Trial",
            "UPSELL",
            "FIXED_AMOUNT",
            5.0,
            6,
            999,
            json.dumps(["Month-to-month", "One year", "Two year"]),
            json.dumps(["DSL", "Fiber optic"]),
            json.dumps(["TechSupport"]),
            "UPSELL_OPPORTUNITY",
            "2026-01-01",
            "2026-12-31",
            "Priority dispatch and remote device troubleshooting for all connected home equipment. $5 monthly billing reduction.",
            "security_bundle_campaigns.csv",
            2,
            1,
            "2026-01-01 00:00:00"
        )
    ]

    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO offers (
            offer_id, offer_name, offer_category, discount_type, discount_value,
            min_tenure, max_tenure, eligible_contract_types, eligible_internet_services,
            required_missing_services, target_segment, validity_start, validity_end,
            terms_and_conditions, source_document, source_page, is_active, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, seed_offers)
    conn.commit()

def _seed_initial_benchmarks(conn: sqlite3.Connection):
    benchmarks = [
        (
            "BENCH-01",
            "Long-term loyalty appreciation discount for customer with over 5 years tenure",
            "LONG_TERM_LOYAL",
            json.dumps(["OFFER-LOYALTY-CREDIT"]),
            "Evaluation query for long-term loyal customer retention offer"
        ),
        (
            "BENCH-02",
            "Security and backup protection addon discount for broadband customer without cybersecurity",
            "UPSELL_OPPORTUNITY",
            json.dumps(["OFFER-SECURITY-BUNDLE-10", "OFFER-TECH-SUPPORT-PLUS"]),
            "Evaluation query for security upsell bundle"
        ),
        (
            "BENCH-03",
            "Contract renewal discount for month-to-month subscriber to lock in annual rate",
            "CONTRACT_RENEWAL_CANDIDATE",
            json.dumps(["OFFER-ANNUAL-LOCKIN-15"]),
            "Evaluation query for contract renewal transition"
        ),
        (
            "BENCH-04",
            "Welcome discount for new customer within first 6 months on streaming services",
            "EARLY_STAGE_CUSTOMER",
            json.dumps(["OFFER-WELCOME-STREAM-25"]),
            "Evaluation query for early onboarding streaming benefit"
        ),
        (
            "BENCH-05",
            "Fiber high speed internet bandwidth upgrade promo with percentage discount",
            "UPSELL_OPPORTUNITY",
            json.dumps(["OFFER-FIBER-BOOST-20"]),
            "Evaluation query for fiber bandwidth boost"
        )
    ]
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO eval_benchmarks (
            benchmark_id, query_text, target_segment, expected_offer_ids, description
        ) VALUES (?, ?, ?, ?, ?)
    """, benchmarks)
    conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
