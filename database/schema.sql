-- SQLite schema for Customer Loyalty & Engagement Assistant
-- assistant.db

CREATE TABLE IF NOT EXISTS offers (
    offer_id TEXT PRIMARY KEY,
    offer_name TEXT NOT NULL,
    offer_category TEXT NOT NULL,
    discount_type TEXT NOT NULL, -- PERCENTAGE, FIXED_AMOUNT, MONTHLY_CREDIT, FREE_TRIAL
    discount_value REAL NOT NULL,
    min_tenure INTEGER DEFAULT 0,
    max_tenure INTEGER DEFAULT 999,
    eligible_contract_types TEXT NOT NULL, -- JSON array of strings e.g. ["Month-to-month", "One year", "Two year"]
    eligible_internet_services TEXT NOT NULL, -- JSON array e.g. ["DSL", "Fiber optic", "No"]
    required_missing_services TEXT, -- JSON array e.g. ["OnlineSecurity"]
    target_segment TEXT NOT NULL, -- e.g. UPSELL_OPPORTUNITY, LONG_TERM_LOYAL, ALL
    validity_start TEXT NOT NULL,
    validity_end TEXT NOT NULL,
    terms_and_conditions TEXT NOT NULL,
    source_document TEXT NOT NULL,
    source_page INTEGER DEFAULT 1,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action_type TEXT NOT NULL, -- RETRIEVAL, VALIDATION, GENERATION, APPROVAL, REJECTION, POLICY_FAILURE, EDIT
    customer_id TEXT,
    user_id TEXT DEFAULT 'system_operator',
    details_json TEXT NOT NULL,
    compliance_status TEXT NOT NULL -- PASSED, FAILED, WARNING, OVERRIDDEN
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    offer_id TEXT,
    segment TEXT NOT NULL,
    original_subject TEXT NOT NULL,
    original_body TEXT NOT NULL,
    final_subject TEXT NOT NULL,
    final_body TEXT NOT NULL,
    reviewer_status TEXT NOT NULL, -- PENDING, APPROVED, REJECTED, EDITED
    reviewer_notes TEXT,
    channel TEXT DEFAULT 'EMAIL',
    confidence_score REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    delivery_status TEXT DEFAULT 'NOT_SENT' -- NOT_SENT, SENT_MOCK, FAILED
);

CREATE TABLE IF NOT EXISTS eval_benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    target_segment TEXT NOT NULL,
    expected_offer_ids TEXT NOT NULL, -- JSON array
    description TEXT
);

CREATE TABLE IF NOT EXISTS uploaded_documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    upload_timestamp TEXT NOT NULL,
    num_chunks INTEGER DEFAULT 0,
    status TEXT DEFAULT 'INDEXED',
    file_size_bytes INTEGER DEFAULT 0,
    checksum_sha256 TEXT UNIQUE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_logs(customer_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_approvals_customer ON approvals(customer_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(reviewer_status);
CREATE INDEX IF NOT EXISTS idx_offers_active ON offers(is_active);
CREATE INDEX IF NOT EXISTS idx_docs_checksum ON uploaded_documents(checksum_sha256);
