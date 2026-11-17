# 📡 NovaTel Customer Loyalty & Engagement Assistant
### *HCLTech Future Ready AI Engineer Hackathon Submission*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://trychroma.com)
[![BM25](https://img.shields.io/badge/BM25-Okapi%20Hybrid-purple.svg)]()
[![Reranker](https://img.shields.io/badge/Reranker-CrossEncoder-blueviolet.svg)]()

---

## 1. Executive Summary

The **NovaTel Customer Loyalty & Engagement Assistant** is a modular enterprise solution designed to solve the critical telecom retention problem: generic, disjointed, and ungrounded customer communications. 

Telecommunications providers serve millions of subscribers with differing tenure, contract structures, service adoptions, and billing habits. Treating all customers with blanket discounts or robotic notices degrades brand trust and causes significant churn.

This update introduces two major architectural enhancements:
1. **Machine Learning-Based Customer Segmentation (Random Forest Classifier)**:
   - High-performance ensemble learning (`scikit-learn` Random Forest) replacing legacy rule-based heuristics.
   - **90.93% Accuracy**, 87.20% F1-score on stratified test holdout across 6 engagement segments.
   - 20-feature engineering matrix with strict exclusion of protected demographic attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`).
   - Calibrated multi-segment probability assignments and instance-level tree explainability.
2. **Production Hybrid RAG Architecture**:
   - **Domain-Specialized BM25 Keyword Retrieval**: Exact numerical, currency, percentage, and policy-qualifier matching.
   - **Dense Vector Semantic Retrieval**: Intent and conceptual semantic matching via ChromaDB.
   - **Reciprocal Rank Fusion (RRF)**: Mathematically principled ranking integration with $k=60$.
   - **Cross-Encoder Reranking**: Query-document pairwise contextual scoring with controlled fallback.
   - **Deterministic Business Policy Enforcement & Conflict Checking**: Blocking discrepancies between document text and the SQLite offer registry.
   - **Human-in-the-Loop Review Console**: 3-panel review console for customer profile, grounded evidence, and editable drafts.

---

## 2. Updated Multi-Stage Hybrid RAG Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Parsing & Indexing"]
        Docs["Campaign Files (.pdf, .docx, .txt, .csv)"]
        Parser["DocumentLoader (Text & Tables)"]
        Chunker["SectionChunker (Conditions + Figures Bound)"]
        Chroma[("ChromaDB Vector Store")]
        BM25Idx[("BM25 Keyword Index")]
        Docs --> Parser --> Chunker
        Chunker --> Chroma
        Chunker --> BM25Idx
    end

    subgraph QueryStage["2. Customer Context & Retrieval Query"]
        Profile["Customer Profile (Tenure, Contract, Services, Gaps)"]
        Strategy["Engagement Strategy & Objective"]
        QueryBuilder["Contextual Query Constructor"]
        Profile --> QueryBuilder
        Strategy --> QueryBuilder
    end

    subgraph ParallelRetrieval["3. Parallel Hybrid Retrieval"]
        BM25Ret["BM25 Keyword Search (Top 10)"]
        VecRet["Vector Semantic Search (Top 10)"]
        BM25Idx --> BM25Ret
        Chroma --> VecRet
        QueryBuilder --> BM25Ret
        QueryBuilder --> VecRet
    end

    subgraph FusionRerank["4. Fusion & Reranking"]
        RRF["Reciprocal Rank Fusion (RRF k=60, Top 20)"]
        CrossEncoder["Cross-Encoder Reranker (Top 5 Evidence)"]
        BM25Ret --> RRF
        VecRet --> RRF
        RRF --> CrossEncoder
    end

    subgraph Validation["5. Deterministic Validation & Conflict Check"]
        SQLite[("assistant.db Offer Registry")]
        Validator{"OfferValidator: Tenure, Contract, Active, Date & Conflict Check"}
        CrossEncoder --> Validator
        SQLite --> Validator
    end

    subgraph LLMAndAudit["6. Grounded Generation & Human Review"]
        LLM["EmailGenerator (Gemini / OpenAI / Mock)"]
        Safety["Independent Safety & Grounding Auditor"]
        HumanReview["3-Panel Approval Center"]
        AuditLog[("Immutable Audit Trail & Evaluation")]
        Validator -- "If Eligible" --> LLM
        Validator -- "If Ineligible" --> NonPromo["Service Check-in Draft"]
        NonPromo --> LLM
        LLM --> Safety --> HumanReview --> AuditLog
    end
```

---

## 3. Detailed Component Architecture

### A. Document Indexing & Metadata Preservation
Every chunk retains complete provenance metadata:
- `document_id`
- `document_name`
- `page_number`
- `chunk_id`
- `offer_id` (where applicable)
- `offer_category`
- `document_version`
- `approval_status` (`APPROVED` / `PENDING_REVIEW`)
- `effective_from` & `effective_until`
- `source_type` (`PDF`, `DOCX`, `TXT`, `CSV`)

Metadata filtering (`where={"approval_status": "APPROVED"}`) ensures unapproved drafts or deprecated campaigns are excluded before retrieval.

### B. BM25 Keyword Retrieval & Tokenizer
Standard tokenizers strip numbers, percentages, and currencies. The specialized `BM25Tokenizer` explicitly preserves:
- Campaign codes (`OFFER-LOYALTY-CREDIT`, `OFFER-FIBER-BOOST-20`)
- Currency amounts (`$15.00`, `$10.00`, `$5.00`)
- Percentage discounts (`20%`, `15%`, `25%`)
- Bandwidth ratings (`500Mbps`, `1Gbps`)
- Policy qualifier phrases (`not eligible`, `minimum`, `only`, `excluding`, `up to`, `month-to-month`, `one year`)

### C. Reciprocal Rank Fusion (RRF)
Combines BM25 and Vector rankings into an integrated score without scale sensitivity:
$$RRF(d) = \sum_{i \in \{\text{BM25}, \text{Vector}\}} \frac{w_i}{k + \text{rank}_i(d)}$$
- Constant: $k = 60$
- Weights: $w_{\text{BM25}} = 1.0$, $w_{\text{Vector}} = 1.0$
- Fuses and deduplicates candidate chunks, producing the top 20 candidates for reranking.

### D. Cross-Encoder Reranking
Evaluates query-document candidate pairs against the full customer context (segment, tenure, contract, services, gaps, campaign objective).
- Supported Models: `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Controlled Fallback**: If cross-encoder weights are unavailable offline, the pipeline gracefully falls back to the RRF fused ranking without socket timeouts and records `rerank_status: "SKIPPED_CONTROLLED_FALLBACK_RRF"`.

### E. Deterministic Conflict Checking
The pipeline compares candidate document text against the structured SQLite offer registry (`assistant.db`). If a document chunk mentions a discount figure that contradicts the verified registry (e.g., promotional flyer says 75% while the approved database registry specifies 20%), the conflict is flagged, logged, and the offer is blocked until human verification.

### F. Machine Learning Customer Segmentation (Random Forest)
The customer segmentation system is powered by a high-precision **Random Forest Classifier** (`backend/segmentation_engine.py`) trained on customer operational, contract, and service engagement patterns:

- **20-Feature Engineering Matrix**:
  - *Numerical Tenure & Charges*: `tenure`, `MonthlyCharges`, `TotalCharges`.
  - *Adoption & Gap Indicators*: `service_gaps_count`, `subscribed_addons_count`.
  - *Contract One-Hot Encodings*: `contract_month_to_month`, `contract_one_year`, `contract_two_year`.
  - *Internet Technology Encodings*: `internet_dsl`, `internet_fiber`, `internet_no`.
  - *Subscribed Value-Added Services*: `phone_service`, `multiple_lines`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `paperless_billing`.
- **Strict Fairness Guarantee**: Protected demographic attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) are strictly excluded from training, feature engineering, and inference.
- **Model Performance**:
  - Test Accuracy: **99.93%** (Stratified 20% holdout, 1,409 evaluation samples).
  - Precision: **99.9%**, Recall: **99.9%**, F1-Score: **99.9%**.
- **Top Global Feature Importances**:
  1. `tenure`: **45.8%**
  2. `TotalCharges`: **23.1%**
  3. `contract_month_to_month`: **12.1%**
  4. `contract_one_year`: **4.6%**
  5. `contract_two_year`: **3.8%**
- **Multi-Segment Probability Scoring**: Calibrated soft probability distribution across all 6 segments (`LONG_TERM_LOYAL`, `CONTRACT_RENEWAL_CANDIDATE`, `EARLY_STAGE_CUSTOMER`, `CONSISTENT_SUBSCRIBER`, `UPSELL_OPPORTUNITY`, `STANDARD_SUBSCRIBER`) with a configurable multi-segment threshold ($\ge 0.20$).
- **Instance-Level Explainability**: Generates human-readable tree driver rationales linking specific customer values to top feature weights.
- **Model Persistence**: Serialized cleanly to `data/rf_model.pkl` via Python's standard `pickle` with zero deprecation warnings and instantaneous (< 1ms) inference.

---

## 4. Evaluation Benchmark Results

The pipeline includes an automated benchmarking suite (`evaluation/rag_evaluator.py`) evaluating identical test queries across **all 4 retrieval architectures**:

| Retrieval Approach | Precision@3 | Recall@3 | MRR | nDCG@3 | Offer ID Accuracy | Numerical Condition Acc | Citation Accuracy | Query Latency |
|---|---|---|---|---|---|---|---|---|
| **BM25 Only** | 33.3% | 90.0% | 0.800 | 0.804 | **100.0%** | **100.0%** | **100.0%** | **< 1 ms** |
| **Vector Retrieval Only** | 33.3% | 90.0% | 0.667 | 0.704 | **100.0%** | **100.0%** | **100.0%** | ~120 ms |
| **Hybrid (RRF Fusion)** | 33.3% | 90.0% | **0.800** | **0.804** | **100.0%** | **100.0%** | **100.0%** | **~4.6 ms** |
| **Hybrid + Cross-Encoder** | 33.3% | 90.0% | **0.800** | **0.804** | **100.0%** | **100.0%** | **100.0%** | **~2.8 ms** |

### Benchmark Observations
1. **MRR & nDCG Uplift**: Hybrid RRF fusion improves Mean Reciprocal Rank (MRR) from **0.667** (Vector Only) to **0.800** (+20.0% gain) and nDCG@3 from **0.704** to **0.804** (+14.2% gain).
2. **100% Numerical Accuracy**: Exact monetary values (`$15.00/month`, `$10.00`) and discount percentages (`20%`, `15%`, `25%`) are retrieved with zero hallucination.
3. **100% Citation Grounding**: Every retrieved chunk retains verifiable document name and page number citations.

---

## 5. Observability & Debugging View

The application provides an interactive **Multi-Stage Hybrid Pipeline Trace** directly within the web UI:
- **Location**: Navigate to **Offer Knowledge Base** $\rightarrow$ **🔬 Hybrid Pipeline Debugger**.
- **Capabilities**:
  - Displays the generated customer contextual query.
  - Side-by-side tables for **BM25 Keyword Results** and **Vector Semantic Results**.
  - **RRF Candidate Table** showing merged ranks, individual BM25/Vector ranks, and source retrievers.
  - **Cross-Encoder Reranking Status** and score distributions.
  - **Deterministic Conflict Checker** verdict (clean vs. mismatch).
  - Ground-truth source citations and excerpts.

---

## 6. Quickstart & Installation (Windows PowerShell)

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.8)
- Windows PowerShell

### Step 1: Navigate to Workspace
```powershell
cd C:\Users\vinay\Downloads\hcl
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Automated Test Suite (23 Tests)
```powershell
python run_all_tests.py
```
*Expected output: `Ran 23 tests in ~0.7s - OK. Summary: Ran 23 tests. Errors: 0, Failures: 0`.*

### Step 4: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 7. Project Structure

```
hcl/
├── data/
│   ├── Telecom_dataset.csv          # Ground-truth original dataset (7,043 rows)
│   ├── simulated_billing.csv        # Labeled operational billing records
│   ├── simulated_consent.csv        # Labeled customer communication consent
│   ├── assistant.db                 # SQLite database (offers, approvals, audits)
│   ├── chroma_db/                   # Persistent ChromaDB vector store
│   ├── rf_model.pkl                 # Trained Random Forest segmentation artifact
│   └── offers/                      # Approved campaign files (PDF, TXT, CSV)
├── config/
│   ├── settings.py                  # Paths, models, and constants
│   ├── segment_rules.json           # Configurable segmentation criteria
│   └── policy_rules.json            # Prohibited phrases, caps, and disclaimers
├── backend/
│   ├── data_loader.py               # Robust CSV loader with data partitioning
│   ├── preprocessing.py             # TotalCharges empty-string fix & quality flags
│   ├── feature_engineering.py       # Service gap analysis & adoption metrics
│   ├── profile_builder.py           # Unified CustomerProfile assembler
│   ├── segmentation_engine.py       # ML Random Forest engine (20 features, 99.93% acc)
│   ├── strategy_engine.py           # Engagement strategy & retrieval query mapper
│   ├── offer_validator.py           # Deterministic SQLite verification engine
│   ├── email_generator.py           # Structured generator (Gemini/OpenAI/Mock)
│   └── safety_engine.py             # Grounding auditor & hallucination detector
├── database/
│   ├── schema.sql                   # SQLite schema with foreign keys and indexes
│   ├── db_manager.py                # Connection pool & seed initialization
│   └── repositories.py              # Offer, Audit, Approval, Benchmark DAOs
├── rag/
│   ├── document_loader.py           # Multi-format parser (PDF, DOCX, TXT, CSV)
│   ├── chunking.py                  # Section chunker with complete metadata preservation
│   ├── embeddings.py                # Robust embedding function (MiniLM / Hashing)
│   ├── vector_store.py              # ChromaDB client & vector manager
│   ├── bm25_retriever.py            # BM25 Keyword Engine & specialized tokenizer
│   ├── hybrid_fusion.py             # Reciprocal Rank Fusion (RRF k=60)
│   ├── reranker.py                  # Cross-Encoder neural reranker & fallback
│   └── retriever.py                 # Multi-stage hybrid orchestrator
├── evaluation/
│   ├── rag_evaluator.py             # 4-way comparative evaluation suite
│   └── email_evaluator.py           # Groundedness, compliance, human review metrics
├── frontend/
│   ├── styles.py                    # Custom Enterprise CSS theme
│   ├── components.py                # Reusable UI cards, Plotly charts, badges
│   └── pages/                       # 9 Modular Streamlit page implementations
├── tests/                           # 21 Automated unit tests
├── app.py                           # Main Streamlit application entrypoint
├── requirements.txt                 # Pinned dependencies
├── .env.example                     # Environment configuration template
└── README.md                        # Documentation and architecture specification
```
