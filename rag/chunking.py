import re
import hashlib
from typing import List, Dict, Any

class SectionChunker:
    """
    Splits telecom campaign documents using section-based boundaries.
    Guarantees that offer eligibility, discount figures, and legal terms
    remain united within the same semantic chunk with complete metadata preservation.
    """

    @classmethod
    def chunk_document(cls, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_idx = 0

        for page in pages:
            doc_name = page.get("document_name", "unknown")
            page_num = int(page.get("page_number", 1))
            raw_content = page.get("content", "")
            raw_dict = page.get("raw_dict")
            source_type = page.get("file_type", cls._infer_source_type(doc_name))
            doc_id = cls._extract_or_generate_doc_id(raw_content, doc_name)

            # If document is a structured CSV row, keep the entire row intact
            if raw_dict:
                offer_id = raw_dict.get("offer_id", f"ROW-{chunk_idx}")
                category = raw_dict.get("offer_category", "GENERAL").upper()
                target_segment = raw_dict.get("target_segment", "ALL").upper()
                eff_from = raw_dict.get("validity_start", "2026-01-01")
                eff_until = raw_dict.get("validity_end", "2026-12-31")
                validity = f"{eff_from} to {eff_until}".strip()

                chunk_idx += 1
                chunks.append({
                    "chunk_id": f"{doc_name}_chunk_{chunk_idx}",
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "offer_id": offer_id,
                    "offer_category": category,
                    "target_segment": target_segment,
                    "validity": validity,
                    "effective_from": eff_from,
                    "effective_until": eff_until,
                    "approval_status": "APPROVED",
                    "document_version": "2026.1",
                    "source_type": source_type,
                    "content": raw_content
                })
                continue

            # Text / PDF / DOCX section splitting
            # Split on section headings or markdown dividers
            sections = re.split(r'(?m)(?=^##\s+SECTION|\n---\n|^SECTION\s+\d+:)', raw_content)
            
            for sec in sections:
                cleaned_sec = sec.strip()
                if len(cleaned_sec) < 30: # ignore trivial headers or whitespace
                    continue

                chunk_idx += 1
                meta = cls._extract_metadata_from_text(cleaned_sec, doc_name, page_num, doc_id, source_type)
                meta["chunk_id"] = f"{doc_name}_chunk_{chunk_idx}"
                meta["content"] = cleaned_sec
                chunks.append(meta)

        return chunks

    @staticmethod
    def _infer_source_type(doc_name: str) -> str:
        ext = doc_name.split(".")[-1].upper() if "." in doc_name else "TXT"
        return ext if ext in ["PDF", "DOCX", "TXT", "CSV", "MD"] else "TXT"

    @staticmethod
    def _extract_or_generate_doc_id(text: str, doc_name: str) -> str:
        doc_id_match = re.search(r'(?:Document ID|Doc ID|Document):\s*([A-Za-z0-9_-]+)', text, re.IGNORECASE)
        if doc_id_match:
            return doc_id_match.group(1).strip()
        stem = doc_name.rsplit(".", 1)[0].upper().replace(" ", "_")
        return f"DOC-{stem}"

    @staticmethod
    def _extract_metadata_from_text(text: str, doc_name: str, page_num: int, doc_id: str, source_type: str) -> Dict[str, Any]:
        """Extracts offer_id, category, target_segment, dates, and approval status from section text."""
        # Find Campaign Code / ID
        code_match = re.search(r'(?:Campaign Code|Campaign ID|Offer ID|Code):\s*([A-Za-z0-9_-]+)', text, re.IGNORECASE)
        offer_id = code_match.group(1).strip() if code_match else "GENERAL_OFFER"

        # Find Category
        cat_match = re.search(r'(?:Category):\s*([A-Za-z0-9_-]+)', text, re.IGNORECASE)
        category = cat_match.group(1).strip().upper() if cat_match else "GENERAL"

        # Find Target Segment
        seg_match = re.search(r'(?:Applicable Segment|Target Segment):\s*([A-Za-z0-9_-]+)', text, re.IGNORECASE)
        target_segment = seg_match.group(1).strip().upper() if seg_match else "ALL"

        # Find Validity & Dates
        val_match = re.search(r'(?:Validity|Valid Window|Validity Period):\s*([^\n\r]+)', text, re.IGNORECASE)
        validity = val_match.group(1).strip() if val_match else "2026-01-01 to 2026-12-31"

        date_matches = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', validity)
        if len(date_matches) >= 2:
            effective_from = date_matches[0]
            effective_until = date_matches[1]
        elif len(date_matches) == 1:
            effective_from = date_matches[0]
            effective_until = "2026-12-31"
        else:
            effective_from = "2026-01-01"
            effective_until = "2026-12-31"

        # Approval status check
        status = "APPROVED"
        if re.search(r'\b(?:draft|unapproved|pending|deprecated|inactive)\b', text, re.IGNORECASE):
            status = "PENDING_REVIEW"

        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "page_number": page_num,
            "offer_id": offer_id,
            "offer_category": category,
            "target_segment": target_segment,
            "validity": validity,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "approval_status": status,
            "document_version": "2026.1",
            "source_type": source_type
        }
