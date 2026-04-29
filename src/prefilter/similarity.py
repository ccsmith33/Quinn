"""S4.2 — Similarity prefilter (ADR-003).

Two-stage check on normalized plaintext:

  Stage 1 — 128-permutation MinHash on 5-word shingles. If estimated Jaccard
            against the most-recent prior `(cik, form_type)` cache entry is
            >= 0.99, reject as exact duplicate (`reason="minhash"`).

  Stage 2 — Per-`(cik, form_type)` TF-IDF cosine. The vectorizer fitted at the
            time of the most-recent prior accepted filing is loaded from disk;
            the new text is transformed and compared (cosine) against the prior
            stored vector. If cosine >= configured threshold (default 0.97 per
            FR-13), reject (`reason="tfidf_cosine"`).

On accept, the checker refits a vectorizer using up to the 5 most recent
same-issuer/form filings (including the current one), persists artifacts
(MinHash blob, TF-IDF vector, fitted vectorizer, normalized text), and writes
a `similarity_cache` row. Retention is capped at 5 per `(cik, form_type)`;
older artifacts and rows are pruned.

Form 4 is excluded — calling `check()` with a Form 4 filing raises
`FormTypeNotSimilarityChecked` to make caller mistakes loud (D-014: Form 4 is
universe-only; PRD §8.3).
"""

from __future__ import annotations

import datetime as _dt
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasketch import MinHash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingestion.normalize import normalize_text
from journal.models import FilingRow, SimilarityCacheRow
from journal.repo import (
    connect,
    get_similarity_cache_for_issuer_form,
    insert_similarity_cache,
)
from observability.log_port import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants (ADR-003 — Stage 1 / Stage 2 parameters)
# ---------------------------------------------------------------------------

_MINHASH_PERMS = 128
_MINHASH_THRESHOLD = 0.99
_SHINGLE_K = 5

_TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=50_000,
)
_RETAIN_PER_KEY = 5
_PICKLE_PROTOCOL = 5

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


class FormTypeNotSimilarityChecked(Exception):
    """Raised when `check()` is called on a form type that bypasses similarity.

    Form 4 is universe-only per PRD §8.3 / D-014. Callers must dispatch on
    `form_type` before invoking the checker; this exception turns silent
    misuse into a loud error.
    """


@dataclass(frozen=True)
class SimilarityDecision:
    decision: Literal["accept", "reject"]
    reason: Literal["minhash", "tfidf_cosine", "pass"]
    score: float | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shingles(normalized: str, k: int = _SHINGLE_K) -> list[str]:
    """k-word shingles over normalized text, with punctuation stripped.

    Punctuation removal makes shingles robust to "earnings, were $1.50" vs
    "earnings were $1.50" — a common cause of MinHash drift on otherwise
    identical text.
    """
    cleaned = _PUNCT_RE.sub(" ", normalized)
    words = cleaned.split()
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _build_minhash(normalized: str) -> MinHash:
    mh = MinHash(num_perm=_MINHASH_PERMS)
    for shingle in _shingles(normalized):
        mh.update(shingle.encode("utf-8"))
    return mh


def _safe_form_dir(form_type: str) -> str:
    """Make form_type safe for use as a directory name."""
    return form_type.replace("/", "_").replace(" ", "_")


def _form_dir(base_dir: Path, cik: int, form_type: str) -> Path:
    return base_dir / str(cik) / _safe_form_dir(form_type)


def _artifact_paths(
    base_dir: Path, cik: int, form_type: str, accession: str
) -> tuple[Path, Path, Path]:
    """Return (vectorizer_path, vector_path, text_path) for a cache entry."""
    d = _form_dir(base_dir, cik, form_type)
    return (
        d / f"{accession}.vectorizer.pkl",
        d / f"{accession}.vector.pkl",
        d / f"{accession}.text.txt",
    )


def _fit_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """Fit a TfidfVectorizer with ADR-003 §"Stage 2" params.

    `min_df=2` is the production setting, but a single-doc corpus (very first
    filing for an issuer/form) needs `min_df=1` to avoid an empty-vocabulary
    error. The threshold for switching is corpus size; once we have >= 2
    documents, `min_df=2` is restored.
    """
    min_df = 2 if len(corpus) >= 2 else 1
    vec = TfidfVectorizer(min_df=min_df, **_TFIDF_PARAMS)
    vec.fit(corpus)
    return vec


# ---------------------------------------------------------------------------
# SimilarityChecker
# ---------------------------------------------------------------------------


class SimilarityChecker:
    """ADR-003 two-stage similarity prefilter.

    Constructor parameters:
      db_path:       path to the SQLite journal (similarity_cache lives here).
      artifact_dir:  filesystem root for vectorizers, vectors, and normalized
                     text snapshots. Production default is the caller's
                     responsibility — this class persists under whatever path
                     it's given.
      threshold:     TF-IDF cosine threshold above which a filing is rejected
                     (FR-13 default 0.97).
    """

    def __init__(
        self,
        *,
        db_path: str,
        artifact_dir: str,
        threshold: float = 0.97,
    ) -> None:
        self.db_path = db_path
        self.artifact_dir = artifact_dir
        self.threshold = threshold
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, filing: FilingRow, raw_text: str) -> SimilarityDecision:
        if filing.form_type == "4":
            raise FormTypeNotSimilarityChecked(
                "Form 4 is universe-only (PRD §8.3 / D-014); similarity check "
                "must not be called on Form 4 filings."
            )

        normalized = normalize_text(raw_text)
        new_minhash = _build_minhash(normalized)

        prior_rows = get_similarity_cache_for_issuer_form(
            self.db_path, filing.cik, filing.form_type
        )

        if not prior_rows:
            self._persist_accept(
                filing=filing,
                normalized=normalized,
                new_minhash=new_minhash,
                prior_rows=prior_rows,
            )
            return SimilarityDecision("accept", "pass", None)

        latest_prior = prior_rows[0]  # ordered fitted_at DESC

        # Stage 1 — MinHash fast path.
        prior_minhash = self._load_minhash(latest_prior.minhash_blob)
        jaccard = float(new_minhash.jaccard(prior_minhash))
        log.debug(
            "similarity stage1",
            extra={
                "cik": filing.cik,
                "form_type": filing.form_type,
                "minhash_jaccard": jaccard,
            },
        )
        if jaccard >= _MINHASH_THRESHOLD:
            return SimilarityDecision("reject", "minhash", jaccard)

        # Stage 2 — TF-IDF cosine main check.
        cosine = self._tfidf_cosine(latest_prior, normalized)
        log.debug(
            "similarity stage2",
            extra={
                "cik": filing.cik,
                "form_type": filing.form_type,
                "tfidf_cosine": cosine,
            },
        )
        if cosine >= self.threshold:
            return SimilarityDecision("reject", "tfidf_cosine", cosine)

        # Accept — persist new artifacts + cache row, refitting the vectorizer.
        self._persist_accept(
            filing=filing,
            normalized=normalized,
            new_minhash=new_minhash,
            prior_rows=prior_rows,
        )
        return SimilarityDecision("accept", "pass", None)

    # ------------------------------------------------------------------
    # Internal — TF-IDF stage
    # ------------------------------------------------------------------

    def _tfidf_cosine(
        self, latest_prior: SimilarityCacheRow, normalized_new: str
    ) -> float:
        vectorizer_path = Path(latest_prior.tfidf_vectorizer_path)
        vector_path = Path(latest_prior.tfidf_vector_path)
        with vectorizer_path.open("rb") as f:
            vectorizer = pickle.load(f)
        with vector_path.open("rb") as f:
            prior_vector = pickle.load(f)

        new_vector = vectorizer.transform([normalized_new])
        sim = float(cosine_similarity(new_vector, prior_vector)[0, 0])
        return sim

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _persist_accept(
        self,
        *,
        filing: FilingRow,
        normalized: str,
        new_minhash: MinHash,
        prior_rows: list[SimilarityCacheRow],
    ) -> None:
        base = Path(self.artifact_dir)
        form_dir = _form_dir(base, filing.cik, filing.form_type)
        form_dir.mkdir(parents=True, exist_ok=True)

        # Build the corpus from up to 4 most-recent prior texts (ordered
        # newest-first → fit on (current + 4 priors) = up to 5 docs total).
        corpus_texts: list[str] = [normalized]
        for prior in prior_rows[: _RETAIN_PER_KEY - 1]:
            text_path = self._text_path_for(prior)
            try:
                corpus_texts.append(text_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                # Defensive: a missing prior text shouldn't crash the
                # accept path; the prior is unavailable for refit but the
                # current filing still gets stored.
                log.warning(
                    "prior similarity text missing — refit corpus shrunk",
                    extra={"missing_path": str(text_path)},
                )

        vectorizer = _fit_vectorizer(corpus_texts)
        new_vector = vectorizer.transform([normalized])

        vectorizer_path, vector_path, text_path = _artifact_paths(
            base, filing.cik, filing.form_type, filing.accession_number
        )
        with vectorizer_path.open("wb") as f:
            pickle.dump(vectorizer, f, protocol=_PICKLE_PROTOCOL)
        with vector_path.open("wb") as f:
            pickle.dump(new_vector, f, protocol=_PICKLE_PROTOCOL)
        text_path.write_text(normalized, encoding="utf-8")

        minhash_blob = pickle.dumps(new_minhash, protocol=_PICKLE_PROTOCOL)

        insert_similarity_cache(
            self.db_path,
            SimilarityCacheRow(
                cik=filing.cik,
                form_type=filing.form_type,
                accession_number=filing.accession_number,
                minhash_blob=minhash_blob,
                tfidf_vectorizer_path=str(vectorizer_path),
                tfidf_vector_path=str(vector_path),
                fitted_at=_dt.datetime.now(),
            ),
        )

        self._prune_old_entries(filing.cik, filing.form_type)

    def _text_path_for(self, row: SimilarityCacheRow) -> Path:
        """Reconstruct the normalized-text artifact path for a cache row."""
        _, _, text_path = _artifact_paths(
            Path(self.artifact_dir),
            row.cik,
            row.form_type,
            row.accession_number,
        )
        return text_path

    def _prune_old_entries(self, cik: int, form_type: str) -> None:
        """Retain only the `_RETAIN_PER_KEY` most recent rows + their files."""
        rows = get_similarity_cache_for_issuer_form(self.db_path, cik, form_type)
        if len(rows) <= _RETAIN_PER_KEY:
            return
        stale = rows[_RETAIN_PER_KEY:]
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for r in stale:
                    conn.execute(
                        "DELETE FROM similarity_cache WHERE cik = ? "
                        "AND form_type = ? AND accession_number = ?",
                        (r.cik, r.form_type, r.accession_number),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        for r in stale:
            for path in (
                Path(r.tfidf_vectorizer_path),
                Path(r.tfidf_vector_path),
                self._text_path_for(r),
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # Internal — MinHash deserialization
    # ------------------------------------------------------------------

    @staticmethod
    def _load_minhash(blob: bytes) -> MinHash:
        return pickle.loads(blob)
