# Quora Semantic Duplicate Detection

Binary classification of question pairs from the Quora Question Pairs dataset: predicting whether two questions are semantically equivalent (`is_duplicate`: 0/1).

## Business Problem

Detecting semantically duplicate questions has direct applications in FAQ deduplication, community Q&A platforms, and semantic search — surfacing the single canonical answer instead of showing users redundant, differently-worded duplicates.

## Data

- Original source: [Kaggle — Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/data)
- Provided directly by the course as an 80/20 stratified split of the original Kaggle dataset (`quora_question_pairs_train.csv.zip`) — see course materials for the download link
- Columns: `question1`, `question2`, `is_duplicate` (plus original dataset indices, kept intentionally)
- Size: 323,429 pairs total — split 80/20 stratified (~258,743 train / ~64,686 validation)
- Class balance: ~37% duplicate, ~63% unique

## Approach

Two heuristic baselines (not counted as one of the required models), followed by four trained models, increasing in complexity:

| # | Model | Tool | Purpose |
|---|-------|------|---------|
| 0.1 | Majority class baseline | — | Reference point |
| 0.2 | TF-IDF cosine similarity + threshold | scikit-learn | Heuristic baseline — a chosen threshold, no learned parameters |
| 1 | Logistic Regression on hand-crafted features | scikit-learn | First trained model — length, word/bigram overlap, and TF-IDF cosine similarity features, standardized (`StandardScaler`) |
| 2 | XGBoost on hand-crafted features | xgboost + Optuna | Mid-complexity model |
| 3 | Fine-tuned Sentence Transformers (`all-MiniLM-L6-v2`) | sentence-transformers | Main model — fine-tuned on training pairs, not just pretrained inference |
| 4 | LLM few-shot classification | Claude API | Fourth model (300–500 examples) |

Primary metric: **log loss** (course requirement — lower is better). Secondary: F1-score, ROC-AUC, Accuracy, Confusion Matrix.

## Results

Pulled from `reports/experiment_table.csv` (most recent run of each model). Accuracy isn't currently computed by the evaluation pipeline (`src/evaluation.py` tracks F1/ROC-AUC/log loss only) — add it there if the course requires it in the final table.

| Model | Log Loss | F1 | ROC-AUC | Accuracy | Notes |
|-------|----------|----|---------|----------|-------|
| 0.1 Majority baseline | 0.6585 | 0.000 | 0.500 | | |
| 0.2 TF-IDF cosine + threshold | 1.1061 | 0.6311 | 0.7344 | | threshold=0.35 |
| 1. Logistic Regression | 0.5227 | 0.6796 | 0.7948 | | hand-crafted features + StandardScaler |
| 2. XGBoost + Optuna | 0.4466 | 0.7027 | 0.8412 | | Optuna-tuned |
| 3. Fine-tuned Sentence Transformers | 0.3483 | 0.8251 | 0.9360 | | full train set, 2 epochs, threshold=0.60 |
| 4. LLM few-shot | 0.3133 | 0.8312 | 0.9405 | | Claude Haiku, 4 hand-picked few-shot examples, threshold=0.5. **Evaluated on a 500-row stratified subsample of val, not the full ~64,686** — not directly comparable in sample size to Models 0-3 |
| **3. Final test set** | **0.3488** | **0.8247** | **0.9360** | | Held-out test (80,858 pairs, never touched before), same threshold=0.60 chosen on train, not re-tuned. Nearly identical to val (0.3483 / 0.8251 / 0.9360) — confirms the model generalizes well |

## Conclusions

**Which model is best, and why.** By raw validation metrics, Model 4 (LLM few-shot) scores highest (log loss 0.313, ROC-AUC 0.941), narrowly ahead of Model 3 (log loss 0.348, ROC-AUC 0.936). But Model 4 was only evaluated on a 500-row subsample (API cost/rate limits), not the full ~64,686-row validation set the other four models were scored on — so the two numbers aren't strictly comparable, and Model 4's edge could partly be sampling variance rather than a real quality gap. Model 3 is the more defensible pick as the primary/production model: it's validated at full scale, runs locally with no per-call API cost or external dependency, and its inference cost is fixed regardless of query volume. Model 4 remains a strong, cheap-to-build alternative (four hand-picked few-shot examples, no training) that's worth validating at full val scale before it could realistically replace Model 3 in production.

Both semantic models (3, 4) clearly outperform the lexical/hand-crafted-feature models (0.2, 1, 2) by a wide margin (log loss 0.35-0.31 vs 0.45-1.1) — the single biggest driver of performance in this project was moving from word-overlap features to learned/pretrained semantic embeddings, not hyperparameter tuning within a given feature set.

**Practical insights (from `06_error_analysis.ipynb`).** Both top models fail in similar ways: pairs on the same topic but with a meaning-changing detail (an opinion question vs. a factual one, a qualifying condition, a changed question word like "how" vs. "why"), genuine synonym/paraphrase duplicates with little shared vocabulary, and typos/abbreviations. A direct same-pair cross-check found no consistent difference between the two models specifically on numeric-detail sensitivity (e.g. "top 10" vs "top 5") — both handle it inconsistently. Manual review of a sample of "errors" also found a substantial share are actually label problems in the source data (~22% for Model 3, ~48% for Model 4, though the latter estimate is noisier given the smaller error pool) — meaning both models' true quality is somewhat understated by the raw metrics above.

**Business application.** Directly supports the use case in [Business Problem](#business-problem): flagging duplicate questions in a Q&A platform, deduplicating FAQ entries, or catching repeated customer inquiries before they're answered twice. Model 3's bi-encoder design (encode once, compare via cosine similarity) is what makes this practical at scale — new-question embeddings can be compared against a precomputed index of existing questions in near-real-time, which a pairwise/cross-encoder or per-call LLM approach can't do cheaply at high query volume.

**Limitations.** (1) Label noise in the underlying Quora dataset affects both training and evaluation — some fraction of "errors" for every model are arguably correct predictions against a wrong label, and this hasn't been corrected at the dataset level, only spot-checked manually on a small sample. (2) Model 4 was evaluated on a much smaller sample than Models 0-3, so its ranking above Model 3 should be treated as provisional, not confirmed. (3) The four Model 4 few-shot examples were hand-picked once and not systematically validated against alternative choices. (4) Model 3 was fine-tuned for only 2 epochs on the full train set — given how cheap training turned out to be (~35 min/3 epochs on a free Colab GPU), there's likely headroom left untried.

**Final test set result.** Model 3, scored once on the held-out test set (`07_test_evaluation.ipynb`, 80,858 pairs, untouched until this point): log loss 0.3488, F1 0.8247, ROC-AUC 0.9360 — nearly identical to its validation numbers (0.3483 / 0.8251 / 0.9360). The threshold (0.60) was fixed from train, not re-tuned on test. This close match confirms the model generalizes well and that the val-based model selection wasn't overfit to val.

**What to improve next.** Score Model 4 on the full validation set (or a much larger subsample) to confirm whether its edge over Model 3 is real. Expand the Model 4 few-shot prompt with a couple of hard examples surfaced by error analysis (not from the evaluated sample itself, to avoid leakage) and re-check whether that closes the numeric-detail inconsistency. Try more epochs / a larger base model for Model 3, now that a full training run is confirmed to be fast and cheap.

### Connection to Signal ML

This project sits in the GenAI/NLP segment of the job market (RAG, embeddings, semantic search — the highest-demand cluster right now), but the underlying math is the same one used in defense/signal roles (radar tracking, ELINT, RF signal classification) that this project's author is targeting long-term. Both problem types reduce to the same pipeline: raw input (text, or a raw signal) → a fixed-length vector representation (a TF-IDF vector, a fine-tuned sentence embedding, or a spectral/waveform feature vector) → a similarity or distance metric in that vector space (cosine similarity here; cosine similarity, correlation, or Mahalanobis distance in signal-processing contexts) → a threshold-based decision.

Model 3's fine-tuning approach in particular makes the connection concrete: training embeddings with `CosineSimilarityLoss` so that semantically equivalent pairs end up close together and non-equivalent pairs end up far apart is the same idea as the metric/contrastive learning used for specific emitter identification (SEI) and multi-target track association (e.g. Kalman Filter / SORT-style data association pipelines). Deciding whether two observations are "the same thing" is a similarity-in-embedding-space problem whether the observation is a sentence pair or a radar return matched across frames. This project was a way to practice that shared foundation on an accessible, well-labeled NLP dataset before applying the same approach to noisier, physics-heavy signal data.

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Get `quora_question_pairs_train.csv.zip` from the course materials (see Data section) and place it in the project root or point notebooks to your path
4. **Update file paths.** The notebooks were developed locally and several cells reference absolute paths on the author's machine (e.g. `/Users/nadiiababanska/Desktop/claude_coowork/ML_final_project/...`) for the dataset CSVs, the saved model in `models/`, and `reports/experiment_table.csv`. These won't resolve on another machine -- search each notebook for `/Users/nadiiababanska/` and replace with the equivalent path in your own clone (or make them relative to the repo root) before running.
5. Run notebooks in `notebooks/` in order (`01_eda.ipynb` → `07_test_evaluation.ipynb`), with one exception: **`04_sentence_transformers.ipynb` is meant to be run in Google Colab with a GPU runtime** (Runtime > Change runtime type > GPU), not locally -- fine-tuning is significantly faster on GPU, and this is what it was actually developed and run on. Upload `quora_with_features.csv` to the Colab session, run the notebook, then download the saved model folder and place it at `models/sentence_transformer_duplicate_model/` in your local clone before continuing to `06_error_analysis.ipynb` / `07_test_evaluation.ipynb` (which both load the model locally). Note: `07_test_evaluation.ipynb` scores the held-out test set and is meant to be run once, with the final chosen model -- re-running it repeatedly defeats the purpose of holding a test set out.
6. To try the live demo: `streamlit run app.py`

Live app: *(Streamlit Cloud link — TBD)*

## Repository Structure

```
quora-semantic-duplicate-detection/
├── README.md
├── requirements.txt
├── app.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_sentence_transformers.ipynb
│   ├── 05_llm_fewshot.ipynb
│   ├── 06_error_analysis.ipynb
│   └── 07_test_evaluation.ipynb
├── src/
│   ├── features.py
│   ├── evaluation.py
│   └── utils.py
└── reports/
    └── experiment_table.csv
```
