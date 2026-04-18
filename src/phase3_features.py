"""
Phase 3 R&D: Positional TF-IDF + Legal Syntactic Features.

Hypothesis: Current pipeline is purely bag-of-words over the whole document. It
loses (a) WHERE a clause appears in the contract and (b) the syntactic structure
of legal modifiers ("shall not", "no more than", "except that", etc.).

Three complementary feature streams:
  1. Global TF-IDF (production baseline, 20K features, word 1-2gram).
  2. Positional TF-IDF: contract is split into 4 quartiles; each gets its own
     5K-feature vectorizer. Concatenated -> 20K positional features.
  3. Legal syntactic features (~30 hand-crafted): modal verb density, negation
     patterns, conditionals, exceptions, temporal quantifiers, liability lingo,
     termination lingo. Computed as scale-independent rates, not raw counts.

All three are stacked horizontally into a single sparse matrix for LightGBM+LR.
No transformer, no heavy deps - pure scipy.sparse + sklearn.
"""
from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# Positional TF-IDF
# =============================================================================

N_SECTIONS = 4
SECTION_MAX_FEATURES = 5_000
SECTION_NGRAM = (1, 2)


def split_into_sections(text: str, n_sections: int = N_SECTIONS) -> List[str]:
    """Cut text into n contiguous sections of roughly equal character length."""
    if not text:
        return [""] * n_sections
    L = len(text)
    boundaries = [int(round(L * i / n_sections)) for i in range(n_sections + 1)]
    return [text[boundaries[i]:boundaries[i + 1]] for i in range(n_sections)]


class PositionalTfidfVectorizer:
    """Stack n independent TF-IDF vectorizers, one per quartile of each document.

    At predict time: split the input doc, each section goes to its own
    vectorizer, then hstack.
    """

    def __init__(
        self,
        n_sections: int = N_SECTIONS,
        max_features_per_section: int = SECTION_MAX_FEATURES,
        ngram_range=SECTION_NGRAM,
    ):
        self.n_sections = n_sections
        self.max_features_per_section = max_features_per_section
        self.ngram_range = ngram_range
        self.vectorizers: List[TfidfVectorizer] = []

    def _section_matrix(self, texts: Sequence[str]) -> List[List[str]]:
        return [split_into_sections(t, self.n_sections) for t in texts]

    def fit(self, texts: Sequence[str]):
        sections = self._section_matrix(texts)
        self.vectorizers = []
        for k in range(self.n_sections):
            vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.ngram_range,
                max_features=self.max_features_per_section,
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
            )
            section_texts = [sec[k] if sec[k].strip() else " " for sec in sections]
            vec.fit(section_texts)
            self.vectorizers.append(vec)
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        sections = self._section_matrix(texts)
        mats = []
        for k, vec in enumerate(self.vectorizers):
            section_texts = [sec[k] if sec[k].strip() else " " for sec in sections]
            mats.append(vec.transform(section_texts))
        return sp.hstack(mats, format="csr")

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @property
    def n_features(self) -> int:
        return sum(len(v.vocabulary_) for v in self.vectorizers)


# =============================================================================
# Legal syntactic features
# =============================================================================

# Patterns designed to survive across paraphrases. Each returns a per-document
# rate (count / doc_len_k_chars) so scale-invariant.
_LEGAL_PATTERNS = {
    # Obligation modals
    "mod_shall": r"\bshall\b",
    "mod_must": r"\bmust\b",
    "mod_may_not": r"\bmay\s+not\b",
    "mod_will_not": r"\bwill\s+not\b",
    "mod_shall_not": r"\bshall\s+not\b",
    # Negations & permission denial
    "neg_no": r"\bno\b(?!\s+(?:later|less|more))",  # plain 'no' (not part of "no later than")
    "neg_not": r"\bnot\b",
    "neg_without": r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?consent\b",
    "neg_prohibit": r"\b(?:prohibit|forbid|preclude|bar|prevent)(?:ed|s|ion)?\b",
    "neg_unauthorized": r"\bunauthori[sz]ed\b",
    # Conditionals & contingency
    "cond_if": r"\bif\b",
    "cond_unless": r"\bunless\b",
    "cond_provided": r"\bprovided\s+that\b",
    "cond_in_event": r"\bin\s+the\s+event\b",
    # Exceptions
    "exc_except": r"\bexcept\b",
    "exc_notwithstanding": r"\bnotwithstanding\b",
    "exc_subject_to": r"\bsubject\s+to\b",
    "exc_save_for": r"\bsave\s+(?:for|as)\b",
    # Temporal quantifiers
    "tq_within_days": r"\bwithin\s+(?:\d+|\w+)\s+(?:business\s+)?(?:days|months|years)\b",
    "tq_no_later": r"\bno\s+later\s+than\b",
    "tq_at_least": r"\bat\s+least\s+(?:\d+|\w+)\b",
    "tq_at_most": r"\b(?:no\s+more\s+than|not\s+more\s+than|up\s+to)\s+(?:\d+|\w+)\b",
    # Liability lingo
    "liab_indemnify": r"\b(?:indemnif(?:y|ies|ied|ication)|hold\s+harmless)\b",
    "liab_limit": r"\b(?:limitation\s+of\s+liability|limit\s+of\s+liability|cap\s+on\s+liability)\b",
    "liab_unlimited": r"\b(?:unlimited\s+liability|no\s+(?:cap|limitation)\s+on\s+liability)\b",
    "liab_extent": r"\bto\s+the\s+(?:fullest\s+)?extent\b",
    "liab_damages": r"\b(?:consequential|incidental|indirect|special|punitive)\s+damages\b",
    # Termination / renewal lingo
    "term_terminate": r"\bterminat(?:e|es|ed|ion|ing)\b",
    "term_expire": r"\bexpir(?:e|es|ed|ation)\b",
    "term_renew": r"\brenew(?:al|ed|ing|s)?\b",
    "term_for_convenience": r"\bfor\s+(?:any|no)\s+reason|\bfor\s+convenience\b",
    # Exclusivity / non-compete lingo
    "excl_exclusive": r"\bexclusive(?:ly)?\b",
    "excl_sole_discretion": r"\bsole\s+discretion\b",
    "excl_non_compete": r"\bnon[- ]?compete\b",
    # IP lingo
    "ip_assign": r"\bassign(?:s|ed|ing|ment|ments)?\b.{0,40}\b(?:intellectual\s+property|ip|patent|copyright|trademark)\b",
    "ip_work_for_hire": r"\bwork\s+(?:made\s+)?for\s+hire\b",
}

LEGAL_FEATURE_NAMES = list(_LEGAL_PATTERNS.keys()) + [
    "doc_len_k_chars",           # document length in thousands of chars (log context)
    "sentence_count_per_k",      # sentences per 1K chars
    "section_count",             # # of all-caps section headers
    "numbered_clause_count_k",   # "(a)", "(b)", "(i)" per 1K chars
    "dollar_mentions_k",         # monetary figures per 1K chars
    "percent_mentions_k",        # percent signs per 1K chars
]

# Precompile regex patterns once
_COMPILED_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in _LEGAL_PATTERNS.items()}
_SENT_PATTERN = re.compile(r"[.!?]\s+")
_SECTION_HEADER = re.compile(r"\n\s*[A-Z][A-Z\s]{4,}[.:]?\s*\n")
_NUMBERED_CLAUSE = re.compile(r"\(\s*(?:[ivx]{1,4}|[a-z]|\d{1,2})\s*\)", re.IGNORECASE)
_DOLLAR_PATTERN = re.compile(r"\$\s*[\d,.]+|\b[\d,]+\s+dollars\b", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"%|\bpercent\b", re.IGNORECASE)


def extract_syntactic_features_one(text: str) -> np.ndarray:
    """Extract ~36 scale-invariant legal syntactic features from one document.

    Counts are normalized per 1,000 characters so feature magnitudes don't depend
    on document length.
    """
    L = len(text)
    L_k = max(1, L) / 1000.0  # length in thousands
    feats = np.zeros(len(LEGAL_FEATURE_NAMES), dtype=np.float32)

    # Regex-based counts per 1K chars
    for i, name in enumerate(LEGAL_FEATURE_NAMES[: len(_LEGAL_PATTERNS)]):
        pat = _COMPILED_PATTERNS[name]
        feats[i] = len(pat.findall(text)) / L_k

    # Structural features (appended after regex features)
    base = len(_LEGAL_PATTERNS)
    feats[base + 0] = L_k                                                 # doc_len_k_chars
    feats[base + 1] = len(_SENT_PATTERN.findall(text)) / L_k              # sentence_count_per_k
    feats[base + 2] = float(len(_SECTION_HEADER.findall(text)))           # section_count
    feats[base + 3] = len(_NUMBERED_CLAUSE.findall(text)) / L_k           # numbered_clause_count_k
    feats[base + 4] = len(_DOLLAR_PATTERN.findall(text)) / L_k            # dollar_mentions_k
    feats[base + 5] = len(_PERCENT_PATTERN.findall(text)) / L_k           # percent_mentions_k
    return feats


def extract_syntactic_features(texts: Sequence[str]) -> np.ndarray:
    """Batch extract over a list of documents, returning (N, n_feat) dense float32."""
    out = np.zeros((len(texts), len(LEGAL_FEATURE_NAMES)), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = extract_syntactic_features_one(t)
    return out


# =============================================================================
# Unified feature builder for Phase 3 experiments
# =============================================================================

def stack_features(
    tfidf_global: sp.csr_matrix,
    tfidf_positional: sp.csr_matrix | None = None,
    syntactic: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Horizontally stack available feature blocks into one sparse matrix."""
    blocks = [tfidf_global]
    if tfidf_positional is not None:
        blocks.append(tfidf_positional)
    if syntactic is not None:
        blocks.append(sp.csr_matrix(syntactic))
    return sp.hstack(blocks, format="csr")
