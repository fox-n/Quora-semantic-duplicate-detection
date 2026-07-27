"""
features.py

Hand-crafted feature engineering 

Text preprocessing:
- preprocess_text(text)                   : lowercase + tokenize + drop English
                                             stop words. Used internally by
                                             word_overlap, bigrams_overlap, and
                                             add_tfidf_cosine_similarity.

Features implemented (validated in Final_project.ipynb via class-split distribution
plots):
- question1_len, question2_len, len_dif   : question lengths and length difference
- common_words, jaccard_similarity,
  word_match_share                        : unigram (word-level) overlap between q1/q2
- bigrams_common, bigram_jaccard_similarity,
  bigram_word_match_share                 : bigram overlap between q1/q2
- cosine_similarity                       : TF-IDF cosine similarity between q1/q2


All row-wise functions expect raw question text (str) and are meant to be
combined with add_*_features() helpers, which take the full DataFrame and
return it with new columns attached.
"""

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer("english")


def preprocess_text(text):
    """Lowercase, tokenize, drop English stop words, and stem the remaining
    tokens. Returns a cleaned string (tokens rejoined with a single space).

    Used before word_overlap, bigrams_overlap, and TF-IDF vectorization --
    without it, stop words like "what"/"is"/"the" inflate common_words,
    jaccard_similarity, and word_match_share between otherwise unrelated
    questions, since those words match regardless of topic. TF-IDF partially
    compensates for this on its own via IDF down-weighting, but the raw
    word/bigram overlap features do not, so this step matters most for those.

    Uses SnowballStemmer (rule-based suffix stripping) instead of
    WordNetLemmatizer + POS-tagging: stemming is context-independent, so the
    same word always maps to the same stem regardless of sentence structure.
    This matters here because POS-tagging short, informal questions is
    unreliable, and inconsistent tags led to inconsistent lemmas -- which hurt
    the exact-match features (jaccard, word_match_share, bigram overlap) that
    depend on identical tokens matching across question pairs.
    """
    tokens = word_tokenize(text.lower())
    words = [stemmer.stem(w) for w in tokens if w not in ENGLISH_STOP_WORDS and w.isalpha()]

    return " ".join(words)


def add_length_features(df, q1_col="question1", q2_col="question2"):
    """Add question1_len, question2_len, len_dif columns to df (in place-safe:
    returns the same df with new columns added).

    Uses the raw (non-preprocessed) text, since length is a property of the
    original question, not its cleaned token set.
    """
    df["question1_len"] = df[q1_col].str.len()
    df["question2_len"] = df[q2_col].str.len()
    df["len_dif"] = df["question1_len"] - df["question2_len"]
    return df


def word_overlap(question1, question2):
    """Measures how much two texts overlap in word composition: how many
    words they share, and in what proportion. Stop words are removed first
    via preprocess_text() so they don't inflate the overlap score.

    Returns (common_words, jaccard_similarity, word_match_share).
    Note: word_match_share tops out at 0.5 (not 1.0) even for identical
    question pairs, since the denominator double-counts shared words.
    """
    q1_words = set(preprocess_text(question1).split())
    q2_words = set(preprocess_text(question2).split())

    common = q1_words & q2_words
    union = q1_words | q2_words
    if len(union) == 0:
        return 0, 0, 0
    common_words = len(common)
    jaccard_similarity = common_words / len(union)
    word_match_share = common_words / (len(q1_words) + len(q2_words))
    return common_words, jaccard_similarity, word_match_share


def add_word_overlap_features(df, q1_col="question1", q2_col="question2"):
    """Apply word_overlap() row-wise and attach common_words,
    jaccard_similarity, word_match_share columns to df."""
    df[["common_words", "jaccard_similarity", "word_match_share"]] = df.apply(
        lambda row: word_overlap(row[q1_col], row[q2_col]), axis=1, result_type="expand"
    )
    return df


def bigrams_overlap(question1, question2):
    """Measures how much the bigrams (consecutive word pairs) of two
    questions overlap -- adds sensitivity to local word order that plain
    word_overlap ignores. Stop words are removed first via preprocess_text()
    before forming bigrams.

    Returns (bigrams_common, bigram_jaccard_similarity, bigram_word_match_share).
    """
    q1_words = preprocess_text(question1).split()
    q2_words = preprocess_text(question2).split()

    q1_bigram = set(zip(q1_words, q1_words[1:]))
    q2_bigram = set(zip(q2_words, q2_words[1:]))

    bigram_common = q1_bigram & q2_bigram
    bigram_union = q1_bigram | q2_bigram
    if len(bigram_union) == 0:
        return 0, 0, 0
    bigrams_common = len(bigram_common)
    bigram_jaccard_similarity = bigrams_common / len(bigram_union)
    bigram_word_match_share = bigrams_common / (len(q1_bigram) + len(q2_bigram))
    return bigrams_common, bigram_jaccard_similarity, bigram_word_match_share


def add_bigram_overlap_features(df, q1_col="question1", q2_col="question2"):
    """Apply bigrams_overlap() row-wise and attach bigrams_common,
    bigram_jaccard_similarity, bigram_word_match_share columns to df."""
    df[["bigrams_common", "bigram_jaccard_similarity", "bigram_word_match_share"]] = df.apply(
        lambda row: bigrams_overlap(row[q1_col], row[q2_col]), axis=1, result_type="expand"
    )
    return df


def add_tfidf_cosine_similarity(df, q1_col="question1", q2_col="question2"):
    """Fit a TfidfVectorizer on the combined q1+q2 corpus (shared vocabulary,
    required so the two vectorized columns live in the same vector space),
    transform question1/question2 separately, and attach a cosine_similarity
    column (pairwise, row-to-row -- not the full N x N similarity matrix).

    Both columns are run through preprocess_text() first (stop words removed)
    before vectorizing. Uses paired_cosine_distances, which returns
    *distance*, hence the 1 - ... to convert to similarity.

    Clipped to [0, 1]: TF-IDF vectors are non-negative, so true cosine
    similarity is mathematically bounded to [0, 1] -- but 1 - distance can
    land a hair outside that range (e.g. -4.44e-16) due to floating-point
    rounding, especially for near-zero vectors (a question that's entirely
    stop words becomes an empty string after preprocess_text, hence a
    zero vector). Downstream, log_loss/roc_auc reject values outside
    [0, 1] outright, so this needs to be clipped here at the source rather
    than patched at the metrics layer.
    """
    q1_clean = df[q1_col].apply(preprocess_text)
    q2_clean = df[q2_col].apply(preprocess_text)

    combined = pd.concat([q1_clean, q2_clean], ignore_index=True)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined)

    vect_q1 = vectorizer.transform(q1_clean)
    vect_q2 = vectorizer.transform(q2_clean)

    cosine_similarity = 1 - paired_cosine_distances(vect_q1, vect_q2)
    df["cosine_similarity"] = cosine_similarity.clip(0, 1)
    return df


def add_all_features(df, q1_col="question1", q2_col="question2"):
    """Convenience wrapper: run the full feature pipeline (length, word
    overlap, bigram overlap, TF-IDF cosine similarity) and return df with
    all engineered columns attached."""
    df = add_length_features(df, q1_col, q2_col)
    df = add_word_overlap_features(df, q1_col, q2_col)
    df = add_bigram_overlap_features(df, q1_col, q2_col)
    df = add_tfidf_cosine_similarity(df, q1_col, q2_col)
    return df
