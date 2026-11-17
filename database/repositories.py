import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from database.db_manager import get_db_connection

class OfferRepository:
    @staticmethod
    def get_all_offers(active_only: bool = True) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM offers"
        if active_only:
            query += " WHERE is_active = 1"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            d = dict(r)
            d["eligible_contract_types"] = json.loads(d["eligible_contract_types"])
            d["eligible_internet_services"] = json.loads(d["eligible_internet_services"])
            d["required_missing_services"] = json.loads(d["required_missing_services"] or "[]")
            results.append(d)
        return results

    @staticmethod
    def get_offer_by_id(offer_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM offers WHERE offer_id = ?", (offer_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        d["eligible_contract_types"] = json.loads(d["eligible_contract_types"])
        d["eligible_internet_services"] = json.loads(d["eligible_internet_services"])
        d["required_missing_services"] = json.loads(d["required_missing_services"] or "[]")
        return d

    @staticmethod
    def add_offer(offer_data: Dict[str, Any]) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO offers (
                offer_id, offer_name, offer_category, discount_type, discount_value,
                min_tenure, max_tenure, eligible_contract_types, eligible_internet_services,
                required_missing_services, target_segment, validity_start, validity_end,
                terms_and_conditions, source_document, source_page, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            offer_data["offer_id"],
            offer_data["offer_name"],
            offer_data["offer_category"],
            offer_data["discount_type"],
            offer_data["discount_value"],
            offer_data.get("min_tenure", 0),
            offer_data.get("max_tenure", 999),
            json.dumps(offer_data.get("eligible_contract_types", ["Month-to-month", "One year", "Two year"])),
            json.dumps(offer_data.get("eligible_internet_services", ["DSL", "Fiber optic", "No"])),
            json.dumps(offer_data.get("required_missing_services", [])),
            offer_data.get("target_segment", "ALL"),
            offer_data.get("validity_start", "2026-01-01"),
            offer_data.get("validity_end", "2026-12-31"),
            offer_data.get("terms_and_conditions", "Terms apply."),
            offer_data.get("source_document", "manual_entry"),
            offer_data.get("source_page", 1),
            offer_data.get("is_active", 1),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
        return True

    @staticmethod
    def update_offer_status(offer_id: str, is_active: bool) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE offers SET is_active = ? WHERE offer_id = ?", (1 if is_active else 0, offer_id))
        conn.commit()
        conn.close()
        return True


class AuditRepository:
    @staticmethod
    def log_action(
        action_type: str,
        customer_id: Optional[str],
        details: Dict[str, Any],
        compliance_status: str = "PASSED",
        user_id: str = "system_operator"
    ) -> int:
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, action_type, customer_id, user_id, details_json, compliance_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, action_type, customer_id, user_id, json.dumps(details), compliance_status))
        conn.commit()
        log_id = cursor.lastrowid
        conn.close()
        return log_id

    @staticmethod
    def get_recent_logs(limit: int = 100, action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        if action_type:
            cursor.execute("SELECT * FROM audit_logs WHERE action_type = ? ORDER BY log_id DESC LIMIT ?", (action_type, limit))
        else:
            cursor.execute("SELECT * FROM audit_logs ORDER BY log_id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = json.loads(d["details_json"])
            except:
                d["details"] = {}
            results.append(d)
        return results

    @staticmethod
    def get_audit_summary() -> Dict[str, int]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT action_type, COUNT(*) as cnt FROM audit_logs GROUP BY action_type")
        summary = {r["action_type"]: r["cnt"] for r in cursor.fetchall()}
        cursor.execute("SELECT compliance_status, COUNT(*) as cnt FROM audit_logs GROUP BY compliance_status")
        for r in cursor.fetchall():
            summary[f"status_{r['compliance_status']}"] = r["cnt"]
        conn.close()
        return summary


class ApprovalRepository:
    @staticmethod
    def create_approval(
        approval_id: str,
        customer_id: str,
        offer_id: Optional[str],
        segment: str,
        subject: str,
        body: str,
        channel: str = "EMAIL",
        status: str = "PENDING",
        reviewer_notes: str = "",
        decided_at: Optional[str] = None,
        confidence_score: float = 0.0
    ) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_decided = decided_at
        if status == "APPROVED" and not final_decided:
            final_decided = now

        cursor.execute("""
            INSERT OR REPLACE INTO approvals (
                approval_id, customer_id, offer_id, segment, original_subject,
                original_body, final_subject, final_body, reviewer_status,
                reviewer_notes, channel, confidence_score, created_at, decided_at, delivery_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NOT_SENT')
        """, (approval_id, customer_id, offer_id, segment, subject, body, subject, body, status, reviewer_notes, channel, float(confidence_score), now, final_decided))
        conn.commit()
        conn.close()
        return True

    @staticmethod
    def auto_approve_high_confidence(threshold: float = 0.70) -> int:
        """
        Auto-approves any pending approval records whose retrieval confidence_score > threshold.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            SELECT approval_id, confidence_score, customer_id FROM approvals
            WHERE reviewer_status = 'PENDING' AND confidence_score > ?
        """, (threshold,))
        eligible = cursor.fetchall()
        audit_entries = []
        for r in eligible:
            appr_id = r["approval_id"]
            conf = float(r["confidence_score"] or 0.0)
            cid = r["customer_id"]
            notes = f"Auto-approved by System: RRF Retrieval Confidence ({conf * 100:.1f}%) exceeds {threshold * 100:.0f}% threshold."
            cursor.execute("""
                UPDATE approvals
                SET reviewer_status = 'APPROVED',
                    reviewer_notes = ?,
                    decided_at = ?
                WHERE approval_id = ?
            """, (notes, now, appr_id))
            audit_entries.append((cid, appr_id, conf))
        conn.commit()
        conn.close()

        # Log audit actions after closing approvals connection to prevent SQLite locking
        for cid, appr_id, conf in audit_entries:
            AuditRepository.log_action(
                action_type="APPROVAL",
                customer_id=cid,
                details={
                    "approval_id": appr_id,
                    "auto_approved": True,
                    "confidence_score": conf,
                    "threshold": threshold,
                    "reason": "RRF retrieval confidence > 70% auto-approval"
                },
                compliance_status="PASSED",
                user_id="system_auto_approval"
            )
        return len(audit_entries)

    @staticmethod
    def update_decision(
        approval_id: str,
        status: str, # APPROVED, REJECTED, EDITED
        final_subject: Optional[str] = None,
        final_body: Optional[str] = None,
        notes: str = "",
        delivery_status: str = "NOT_SENT"
    ) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # If subject/body not provided, keep existing
        cursor.execute("SELECT final_subject, final_body FROM approvals WHERE approval_id = ?", (approval_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
            
        subj = final_subject if final_subject is not None else row["final_subject"]
        bdy = final_body if final_body is not None else row["final_body"]

        cursor.execute("""
            UPDATE approvals
            SET reviewer_status = ?,
                final_subject = ?,
                final_body = ?,
                reviewer_notes = ?,
                decided_at = ?,
                delivery_status = ?
            WHERE approval_id = ?
        """, (status, subj, bdy, notes, now, delivery_status, approval_id))
        conn.commit()
        conn.close()
        return True

    @staticmethod
    def get_approvals(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        if status_filter and status_filter != "ALL":
            cursor.execute("SELECT * FROM approvals WHERE reviewer_status = ? ORDER BY created_at DESC", (status_filter,))
        else:
            cursor.execute("SELECT * FROM approvals ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_approval_by_id(approval_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None


class BenchmarkRepository:
    @staticmethod
    def get_all_benchmarks() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM eval_benchmarks")
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["expected_offer_ids"] = json.loads(d["expected_offer_ids"])
            results.append(d)
        return results


class DocumentRepository:
    """
    Tracks uploaded documents, file sizes, and cryptographic SHA-256 checksums
    to strictly prevent redundant/duplicate document ingestion.
    """
    @staticmethod
    def get_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploaded_documents WHERE checksum_sha256 = ?", (checksum,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def record_document(
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        checksum: str,
        num_chunks: int = 0
    ) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT OR REPLACE INTO uploaded_documents (
                doc_id, filename, file_type, upload_timestamp, num_chunks, status, file_size_bytes, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, 'INDEXED', ?, ?)
        """, (doc_id, filename, file_type, now, num_chunks, file_size_bytes, checksum))
        conn.commit()
        conn.close()
        return True

    @staticmethod
    def get_all_documents() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploaded_documents ORDER BY upload_timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def delete_document(doc_id: str) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM uploaded_documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        return True
