"""
utils.py

General helper functions shared across notebooks.

- plot_class_split_hist(...)  : overlaid density histogram of a feature,
                                 split by is_duplicate == 1 vs 0 -- the
                                 single most repeated plotting pattern in
                                 the EDA notebook (question lengths, word
                                 overlap features, bigram features, cosine
                                 similarity all use this same shape of plot)
- find_mismatched_cases(...)  : filter rows where a similarity score
                                 disagrees with the is_duplicate label
                                 (e.g. high similarity but not-duplicate,
                                 or low similarity but duplicate) -- used
                                 for "hard case" examples in EDA, and
                                 reusable later in 06_error_analysis.ipynb
                                 once real model predictions exist
"""

import matplotlib.pyplot as plt


def plot_class_split_hist(duplicates, not_duplicates, column, title=None, bins=35):
    """Plot an overlaid, density-normalized histogram of `column`, split by
    class (Duplicate vs Not duplicate).

    `duplicates` / `not_duplicates` are pre-filtered DataFrames (e.g.
    raw_df[raw_df['is_duplicate'] == 1] / == 0). density=True is used
    because the classes are imbalanced (~63/37) -- without it, the "Not
    duplicate" histogram would always look taller purely due to class size,
    making the shapes hard to compare.

    Does not call plt.figure()/plt.subplot()/plt.show() itself, so it can
    be dropped into an existing subplot grid (call plt.subplot(...) before
    calling this, and plt.tight_layout()/plt.show() after the last one).
    """
    plt.hist(duplicates[column], alpha=0.5, label='Duplicate', bins=bins, color='aqua', density=True)
    plt.hist(not_duplicates[column], alpha=0.5, label='Not duplicate', bins=bins, color='blue', density=True)
    plt.legend()
    plt.title(title or f'{column} distribution')


def find_mismatched_cases(df, score_col, threshold, direction, target_label, label_col='is_duplicate'):
    """Find rows where a similarity score disagrees with the true label --
    the "hard cases" that a simple lexical/similarity feature would get
    wrong.

    direction='above'  -> score_col > threshold  (e.g. high similarity but
                           target_label == 0: looks like a duplicate, isn't)
    direction='below'  -> score_col < threshold  (e.g. low similarity but
                           target_label == 1: is a duplicate, doesn't look
                           like one -- paraphrasing/synonyms case)

    Returns the filtered DataFrame (unsorted, caller can .head(n)).
    """
    if direction == 'above':
        mask = df[score_col] > threshold
    elif direction == 'below':
        mask = df[score_col] < threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    return df[mask & (df[label_col] == target_label)]
