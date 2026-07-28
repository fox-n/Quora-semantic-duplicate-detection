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

## Conclusions

*(to be filled in after error analysis)*

### Connection to Signal ML

*(paragraph connecting embeddings + cosine similarity in NLP semantic matching to signal classification — relevant to defense/signal tech applications)*

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Get `quora_question_pairs_train.csv.zip` from the course materials (see Data section) and place it in the project root or point notebooks to your path
4. Run notebooks in `notebooks/` in order (`01_eda.ipynb` → `06_error_analysis.ipynb`)
5. To try the live demo: `streamlit run app.py`

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
│   └── 06_error_analysis.ipynb
├── src/
│   ├── features.py
│   ├── evaluation.py
│   └── utils.py
└── reports/
    └── experiment_table.csv
```
