# Credit Card Fraud Detection

A reproducible, presentation-ready binary-classification project using a modest TensorFlow/Keras neural network and Kaggle's public Credit Card Fraud Detection dataset. The notebook checks the downloaded data, explores the severe class imbalance, uses train-only preprocessing, selects a decision threshold on validation data, and evaluates the held-out test set once.

## Quick path

1. From this project root, create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Start Jupyter from this project root and run the notebook from top to bottom:

   ```bash
   jupyter lab credit_card_fraud_detection.ipynb
   ```

   The first data cell uses KaggleHub to download the latest public version of `creditcard.csv` and copy it to `data/creditcard.csv`. Internet access is required. Kaggle authentication may be needed if Kaggle requires user consent for this public dataset. Never paste tokens or credentials into the notebook or project.

3. **Optional reference code only:** `kaggle kernels pull` downloads notebook/source files; it does not obtain this dataset and is not needed to download the CSV. The Kaggle CLI is not a project dependency. To retrieve the optional reference notebook, install and configure that CLI separately, then run:

   ```bash
   kaggle kernels pull gpreda/credit-card-fraud-detection-predictive-models -p references
   ```

## Data source and provenance

The notebook requests the exact `creditcard.csv` file from Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with KaggleHub (`mlg-ulb/creditcardfraud`, `path="creditcard.csv"`). KaggleHub returns a local cached file path; the notebook checks that it exists and is a file, then stages it at `data/creditcard.csv` for the existing notebook workflow and course submission. The request is made when the notebook runs so KaggleHub can provide the latest public dataset version.

Internet access is required. Authentication is only needed if Kaggle requires user consent for this public resource; do not paste credentials into notebook cells or project files.

## Project contents and scope

| File | Purpose |
|---|---|
| `credit_card_fraud_detection.ipynb` | Data checks, exploratory analysis, stratified splitting, train-only scaling, neural-network training, validation threshold selection, and final test evaluation. |
| `requirements.txt` | Python packages for the notebook, Jupyter, and KaggleHub data download. |
| `.gitignore` | Excludes Python/Jupyter caches, the virtual environment, and the downloaded CSV. |

The project uses Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and treats `Class=1` as fraud and `Class=0` as a legitimate transaction. The notebook reports the loaded CSV's observed dimensions and checks required columns, data types, missing values, duplicate rows, and class values; it does not enforce a hardcoded row count. It reports Average Precision (PR-AUC), ROC-AUC as a secondary metric, precision, recall, F1, a confusion matrix, and a classification report. Accuracy is not used alone because fraud is rare.

The neural network uses two modest hidden layers and training-only class weights. `StandardScaler` is fit on the training partition only. Early stopping monitors validation PR-AUC; the demonstration threshold is chosen by maximizing validation F1. These are modeling choices, not guarantees: class weighting does not fix every imbalance issue, and a threshold selected on one validation sample may not match real-world operating costs.

## Submission and video checklist

**Deadline: 15 November 2026, 23:59.** The assignment is available only to students who passed all five self-assessments on the first attempt and achieved at least 72/85. This is an eligibility prerequisite; this project cannot validate it.

The video must be no longer than **15 minutes**, and the student must be visible (not voice-over only). A suggested outline follows the six rubric point blocks (2 + 3 + 4 + 2 + 3 + 1 = 15 points); use the assignment rubric's own wording when presenting:

| Suggested segment | Points | What to show |
|---|---:|---|
| Problem and dataset | 2 | Fraud-detection objective, public Kaggle source, and target definition. |
| Exploratory analysis | 3 | Data-quality checks, class balance, and relevant transaction plots. |
| Preprocessing and neural network | 4 | Stratified split, train-only scaling, class weighting, model, and early stopping. |
| Threshold and evaluation | 2 | Validation-only threshold choice and held-out test metrics, including recall and PR-AUC. |
| Interpretation and limitations | 3 | Explain the observed results from the executed notebook; discuss tradeoffs and limits without claiming future-time performance. |
| Reproducibility and delivery | 1 | Show how to obtain the data and run the project; confirm submission permissions. |

Before delivery, confirm that the submission includes:

- [ ] The presentation video, within the 15-minute cap, with the student visible.
- [ ] The `creditcard.csv` dataset, even though `.gitignore` excludes it from version control.
- [ ] The Python code/notebook, with download and view permissions enabled for the intended reviewers.
- [ ] Actual notebook outputs and a conclusion based on those outputs; this starter notebook contains no precomputed or fabricated results.

## Known limitations

- The data covers a short historical interval of approximately two days in 2013; it does not establish current fraud behavior.
- `V1`–`V28` are anonymized PCA-derived features, which limits semantic explanations of individual signals.
- A random stratified split measures performance under a similar sampled distribution; it does not establish generalization to future transactions or temporal drift.
- The validation-selected F1 threshold is a demonstration choice. Real deployment requires an explicit cost policy, operational constraints, and separate monitoring.
- The CSV is ignored by Git to avoid accidentally committing a large data file, but it **must still be included in the final course submission**.

## Professional Python/Jupyter version

The existing starter project and notebook remain intact. For an additive, package-based walkthrough with a chronological train/validation/test split, leakage-safe preprocessing, model benchmarks, offline unit tests, and a notebook aligned to the course rubric, see [`professional/README.md`](professional/README.md).
