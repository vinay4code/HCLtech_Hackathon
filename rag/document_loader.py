import os
import csv
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

class DocumentLoader:
    """
    Extracts text, tables, and page metadata from PDF, DOCX, TXT, and CSV files.
    Preserves document structure and page references.
    """

    @classmethod
    def load_document(cls, file_path: Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return cls._load_pdf(file_path)
        elif ext in [".txt", ".md"]:
            return cls._load_txt(file_path)
        elif ext == ".csv":
            return cls._load_csv(file_path)
        elif ext == ".docx":
            return cls._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")

    @classmethod
    def _load_pdf(cls, file_path: Path) -> List[Dict[str, Any]]:
        pages = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append({
                        "document_name": file_path.name,
                        "page_number": page_num + 1,
                        "content": text.strip(),
                        "file_type": "PDF"
                    })
            doc.close()
        except ImportError:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "document_name": file_path.name,
                        "page_number": page_num + 1,
                        "content": text.strip(),
                        "file_type": "PDF"
                    })
        return pages

    @classmethod
    def _load_txt(cls, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{
            "document_name": file_path.name,
            "page_number": 1,
            "content": content.strip(),
            "file_type": "TXT"
        }]

    @classmethod
    def _load_csv(cls, file_path: Path) -> List[Dict[str, Any]]:
        pages = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for idx, row in enumerate(rows):
            # Format row as key-value text chunk
            row_text = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
            pages.append({
                "document_name": file_path.name,
                "page_number": idx + 1,
                "content": row_text,
                "raw_dict": row,
                "file_type": "CSV"
            })
        return pages

    @classmethod
    def _load_docx(cls, file_path: Path) -> List[Dict[str, Any]]:
        """Extracts text from DOCX archive natively without external library dependency."""
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for node in tree.iter():
                if node.tag.endswith("}p"):
                    texts = [child.text for child in node.iter() if child.tag.endswith("}t") and child.text]
                    if texts:
                        paragraphs.append("".join(texts))
            full_text = "\n\n".join(paragraphs)
            return [{
                "document_name": file_path.name,
                "page_number": 1,
                "content": full_text.strip(),
                "file_type": "DOCX"
            }]
        except Exception as e:
            return [{
                "document_name": file_path.name,
                "page_number": 1,
                "content": f"[DOCX parsing error: {e}]",
                "file_type": "DOCX"
            }]
