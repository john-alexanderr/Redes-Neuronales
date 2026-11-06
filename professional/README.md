# Professional Credit Card Fraud Detection Walkthrough

This additive Python/Jupyter version keeps the existing project intact and provides a reproducible, leakage-aware treatment of the highly imbalanced Kaggle card-fraud dataset. The required neural network remains the primary assignment model; two small baselines explain why accuracy alone is not enough and provide context without introducing an algorithm zoo.

## Quick path: Docker on Windows

From this directory in PowerShell, build the image and start JupyterLab:

```powershell
docker compose up --build --detach
```

Open `http://localhost:8888`. To see the generated Jupyter access URL and token, follow the service logs:

```powershell
docker compose logs --follow notebook
```

Press Ctrl+C to stop following the logs; the service keeps running. Run the unit tests in a one-off container:

```powershell
docker compose run --rm notebook python -m pytest -q
```

Stop the Jupyter service when finished:

```powershell
docker compose down
```

The image installs the project and its `dev` extra, including pytest. The Compose service mounts this directory into `/workspace`; the image build excludes the local dataset and Kaggle credential files.

## Local Python setup

From this directory, install the project and its offline test dependency, then launch the walkthrough:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
jupyter lab notebooks/credit_card_fraud_detection_walkthrough.ipynb
```

The notebook downloads `creditcard.csv` to `professional/data/creditcard.csv` only when that local file is absent. KaggleHub needs internet access. The public dataset may be available without authentication; if KaggleHub requires consent or authentication, provide it in the host shell session before starting Compose. Compose passes through `KAGGLE_API_TOKEN` only when it is set in that session. Do not put credentials in project files, Compose configuration, or the image. The CSV is ignored by Git and excluded from the image build context.

## What the walkthrough does

| Stage | Practice |
|---|---|
| Data | Validate `Time`, `V1`–`V28`, `Amount`, and binary `Class`; show observed data-quality facts without hardcoded row counts. |
| Split | Use `Time` alone for an approximately 65% earliest / 15% middle / 20% latest split, keeping identical timestamps together and reporting realized proportions. |
| EDA | Restrict label-conditioned counts, plots, and correlations to training data. |
| Preprocessing | Fit the MLP's `StandardScaler` on training features only; the logistic-regression benchmark uses a train-fitted scaler in its own pipeline. |
| Models | Compare a most-frequent `DummyClassifier`, balanced `LogisticRegression`, and the required Keras MLP (64 ReLU → Dropout 0.2 → 32 ReLU → Dropout 0.2 → sigmoid). |
| Selection | Use scikit-learn Average Precision for validation comparison. Compute class weights from training labels and use validation PR-AUC for Keras early stopping. Choose the logistic and MLP demonstration F1 thresholds on validation only. |
| Final report | Freeze model and thresholds before test evaluation. Report Average Precision, secondary ROC-AUC, precision, recall, F1, confusion matrices, and the majority baseline's accuracy and fraud recall. |

Average Precision is scikit-learn's step-wise precision-recall summary; it is not described as trapezoidal PR-AUC. The Keras PR-AUC metric is used for early stopping and is not substituted for the final Average Precision report. F1 is a demonstration threshold objective, not a business cost function. A test partition with only one class has undefined ROC-AUC; the evaluation utility returns `NaN` rather than inventing a value.

Chronological evaluation better represents scoring later transactions than random stratification, but this dataset spans only about two days in 2013. It cannot establish robust future-time performance or model current fraud patterns. The anonymized `V1`–`V28` columns are PCA-derived, limiting semantic explanations. Class weighting increases the loss contribution of rare fraud examples without creating new information; threshold selection still trades false alarms against missed fraud and requires a real cost policy for deployment.

## Rubric and course delivery

The notebook's opening map connects its sections to the six course rubric blocks: problem and dataset (2 points), exploratory analysis (3), preprocessing and neural network (4), threshold and evaluation (2), interpretation and limitations (3), and reproducibility and delivery (1). Use the actual rubric wording in the presentation and write conclusions only from outputs produced by your run.

- **Eligibility:** the assignment is for students who passed all five self-assessments on the first attempt and earned at least 72/85.
- **Deadline:** 15 November 2026 at 23:59.
- **Video:** no longer than 15 minutes, with the student visible (not voice-over only).
- **Delivery:** include the dataset CSV despite its Git ignore rule, provide code/notebook view access to intended reviewers, and submit actual notebook outputs and an evidence-based conclusion. Do not commit the downloaded CSV.

## Project layout

```text
professional/
├── notebooks/credit_card_fraud_detection_walkthrough.ipynb
├── src/fraud_detection/
└── tests/
```

The tests use small synthetic inputs and do not download data, import TensorFlow, or train a model.
