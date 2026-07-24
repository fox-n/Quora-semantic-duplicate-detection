"""
evaluation.py

Shared evaluation utilities used across all model notebooks (baseline,
XGBoost, fine-tuned Sentence Transformers, LLM few-shot) to keep metrics
consistent.

Primary metric: log loss (course requirement — lower is better).
Secondary metrics: F1-score (binary), ROC-AUC, Confusion Matrix.

Implemented (extracted from repeated notebook code in Model 0 / Model 1
sections):
- compute_metrics(...)            : f1 / roc_auc / log_loss -> dict, printed
                                     and ready for append_to_experiment_table().
                                     Also records train_time_sec and
                                     train_f1/train_roc_auc/train_log_loss
                                     when train_score/train_time are passed in.
- append_to_experiment_table(...) : append one model's results as a row to
                                     the running experiment CSV (creates the
                                     file with a header on the first call)
- plot_confusion_matrix(...)      : normalized confusion matrix heatmap,
                                     labeled no/yes (not duplicate / duplicate)
- train_and_evaluate(...)         : full pipeline -- fit (timed), predict on
                                     train + validation, metrics for both,
                                     confusion matrix plot, log to experiment
                                     table. Auto-captures model.get_params()
                                     if params isn't passed explicitly. For
                                     "official" model runs (one row per model
                                     in the report), not for hyperparameter
                                     search loops.
- quick_log_loss(...)             : fit + predict_proba + log_loss only, no
                                     printing/plotting/logging. Meant for use
                                     inside an Optuna objective function,
                                     where the same model gets fit 50+ times
                                     and none of that should hit stdout or
                                     the CSV.

Planned (not yet implemented):
- plot_roc_curve(y_true, y_proba)

Reminder: the held-out test set should only be scored once, at the end,
with the best-performing model.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score


def _score(y_true, y_pred, y_score):
    """Compute the f1/roc_auc/log_loss triple for one dataset. Shared helper
    used by train_and_evaluate() to score both the train and validation sets
    with identical logic."""
    return {
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score),
        'log_loss': log_loss(y_true, y_score),
    }


def compute_metrics(y_true, y_pred, y_score, model_name, threshold=None, params=None,
                     train_score=None, train_time=None):
    """Compute the standard metric set for a model and package it as a dict
    ready for append_to_experiment_table(). Also prints each metric.

    y_pred      : hard 0/1 predictions -- used for f1 (needs an actual decision)
    y_score     : continuous score/probability -- used for roc_auc and log_loss.
                  Pass the raw score (e.g. cosine_similarity, or a model's
                  predict_proba output), NOT y_pred -- these two metrics lose
                  their meaning if given already-thresholded 0/1 values.
    params      : optional dict of hyperparameters to record alongside the
                  metrics (e.g. study.best_params from Optuna) -- left as None
                  for models that don't have tunable hyperparameters (baseline,
                  majority class).
    train_score : optional dict {'f1':..., 'roc_auc':..., 'log_loss':...}
                  computed on the training set. Recorded alongside the
                  validation metrics so a train/validation gap (overfitting)
                  is visible directly in the experiment table, not just
                  inferred later.
    train_time  : optional float, seconds spent in model.fit() (wall clock).
    """
    results = {
        'model': model_name,
        'threshold': threshold,
        'params': params,
        'train_time_sec': train_time,
        'train_f1': train_score['f1'] if train_score else None,
        'train_roc_auc': train_score['roc_auc'] if train_score else None,
        'train_log_loss': train_score['log_loss'] if train_score else None,
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score),
        'log_loss': log_loss(y_true, y_score),
    }
    for key, value in results.items():
        print(f'{key}: {value}\n')
    return results


def append_to_experiment_table(results, path='experiment_table.csv'):
    """Append one experiment's results (a dict from compute_metrics) as a
    row to the running experiment table CSV.

    Handles both cases automatically: creates the file with a header if it
    doesn't exist yet, or appends without a header if it does -- so this
    same call works whether it's the first model or the fifth, no need to
    branch manually each time.
    """
    file_exists = os.path.exists(path)
    pd.DataFrame([results]).to_csv(
        path, mode='a' if file_exists else 'w', header=not file_exists, index=False
    )


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot a row-normalized confusion matrix heatmap for a model, labeled
    'no'/'yes' (not duplicate / duplicate)."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.show()


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name, threshold=None, params=None):
    """Full pipeline for one "official" model run: fit -> predict ->
    compute_metrics -> plot_confusion_matrix -> append_to_experiment_table.

    Assumes a binary sklearn-API classifier (has .fit/.predict/.predict_proba).

    params : optional dict of hyperparameters to record in the experiment
             table (e.g. study.best_params from Optuna). If not given, falls
             back to model.get_params(), so the table always records what the
             model was actually configured with -- even for a plain
             XGBClassifier() call with no explicit tuning.
    Also times model.fit() (train_time_sec) and scores the model on the
    training set (train_f1/train_roc_auc/train_log_loss) so a train/validation
    gap is visible directly in the table, not just inferred.

    Returns (fitted_model, results_dict) so the model itself is still
    available afterwards (e.g. for feature_importances_).
    """
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    if params is None:
        params = model.get_params()

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_score = _score(y_train, y_train_pred, y_train_prob)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = compute_metrics(
        y_val, y_pred, y_prob, model_name, threshold, params,
        train_score=train_score, train_time=train_time,
    )
    plot_confusion_matrix(y_val, y_pred, model_name)
    append_to_experiment_table(results)
    return model, results


def quick_log_loss(model, X_train, y_train, X_val, y_val):
    """Fit + predict_proba + log_loss only -- no printing, no plotting, no
    CSV writes. Use this inside an Optuna objective function (or any other
    hyperparameter search loop), where the model gets fit many times and
    only the final best trial's result should go through the full
    train_and_evaluate() pipeline above."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, y_prob)
