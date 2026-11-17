import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter

class BM25Tokenizer:
    """
    Domain-specialized tokenizer for telecom policy, campaign, and offer documents.
    Preserves exact identifiers, percentages, currency, speeds, and qualifier terms.
    """
    QUALIFIER_PHRASES = [
        "not eligible",
        "month-to-month",
        "one year",
        "two year",
        "up to",
        "fiber optic",
        "dsl"
    ]

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        if not text:
            return []

        text_lower = text.lower()
        tokens = []

        # 1. Extract protected phrases and replace with hyphens
        for phrase in cls.QUALIFIER_PHRASES:
            if phrase in text_lower:
                token_form = phrase.replace(" ", "_")
                # Count occurrences and add
                count = text_lower.count(phrase)
                tokens.extend([token_form] * count)
                # mask to prevent re-tokenization
                text_lower = text_lower.replace(phrase, " ")

        # 2. Regex matching currency, percentages, speeds, hyphenated IDs, and terms
        # Patterns:
        # Currency: \$\d+(?:\.\d+)?
        # Percentage: \d+(?:\.\d+)?%
        # Speed: \d+(?:mbps|gbps)
        # Identifiers: [a-z0-9]+(?:[-_][a-z0-9]+)+
        # Words: [a-z0-9]+
        pattern = re.compile(
            r'(\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%|\d+(?:mbps|gbps)|[a-z0-9]+(?:[-_][a-z0-9]+)+|[a-z0-9]+)',
            re.IGNORECASE
        )
        matches = pattern.findall(text_lower)
        for m in matches:
            t = m.lower().strip()
            if len(t) > 0 and t not in ["a", "an", "the", "and", "or", "of", "in", "to", "for", "with"]:
                tokens.append(t)

        return tokens


class BM25Retriever:
    """
    Okapi BM25 Keyword Retrieval Engine.
    Self-contained, fast, and deterministic with support for metadata filtering.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Counter] = []
        self.is_indexed: bool = False

    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Indexes a collection of document chunks.
        Every chunk must contain 'chunk_id', 'content', and optional 'metadata'.
        """
        self.corpus_chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        N = len(chunks)
        if N == 0:
            self.is_indexed = False
            return 0

        # Tokenize each document
        for c in chunks:
            text = c.get("content", "")
            tokens = BM25Tokenizer.tokenize(text)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = sum(self.doc_lengths) / max(1, N)

        # Compute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

        self.is_indexed = True
        return N

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates BM25 scores for all indexed chunks and returns top_k matching chunks.
        """
        if not self.is_indexed or not self.corpus_chunks:
            return []

        query_tokens = BM25Tokenizer.tokenize(query_text)
        if not query_tokens:
            return []

        scores = []
        for idx, (tf, doc_len) in enumerate(zip(self.doc_term_freqs, self.doc_lengths)):
            chunk = self.corpus_chunks[idx]

            # Apply metadata filter if provided
            if metadata_filter:
                meta = chunk.get("metadata", chunk)
                mismatch = False
                for k, v in metadata_filter.items():
                    if meta.get(k) != v and v != "ALL":
                        mismatch = True
                        break
                if mismatch:
                    continue

            # Calculate BM25 score
            score = 0.0
            for term in query_tokens:
                if term in tf:
                    term_tf = tf[term]
                    term_idf = self.idf.get(term, 0.0)
                    # Standard BM25 formula
                    numerator = term_tf * (self.k1 + 1.0)
                    denominator = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                    score += term_idf * (numerator / denominator)

            if score > 0.0:
                scores.append((idx, score))

        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            c = self.corpus_chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "content": c.get("content", ""),
                "metadata": c.get("metadata", c),
                "bm25_score": round(float(score), 4),
                "bm25_rank": rank
            })

        return results
