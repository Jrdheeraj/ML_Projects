# Customer Churn Prediction

Production-level ML system for predicting telecom customer churn with modular training and inference pipelines, model selection and tuning, robust logging, custom exception handling, and FastAPI serving.

## Business Problem

Telecom operators lose recurring revenue when customers churn. This project predicts churn risk at customer level and returns both class label and probability to support retention actions.

## Architecture

```text
customer_churn_prediction/
  data/
    raw/
    processed/
  notebooks/
  src/
    components/
      data_ingestion.py
      preprocessing.py
      feature_engineering.py
      model_trainer.py
      model_evaluation.py
    pipelines/
      training_pipeline.py
      prediction_pipeline.py
    utils/
      io_utils.py
      model_utils.py
    config/
      configuration.py
    logger/
      logging.py
    exception/
      custom_exception.py
  models/
  artifacts/
  api/
    main.py
  tests/
```

## ML Approach

1. Data ingestion and reproducible train/test split.
2. Feature engineering for churn-related telecom patterns.
3. Preprocessing:
   - Missing value handling
   - One-hot encoding
   - Standard scaling for numeric columns
4. Feature selection using model-based importance.
5. Model candidates:
   - Logistic Regression
   - Random Forest
   - XGBoost (if installed)
6. Hyperparameter tuning via `RandomizedSearchCV`.
7. Evaluation metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
8. Best model and metrics persisted for deployment.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place your telecom churn CSV in:

`data/raw/telco_churn.csv`

A realistic sample dataset is included in that location so the project runs without any external dependency.

Expected target column: `Churn` (values `Yes/No` or `1/0`).

Environment overrides:

- `RAW_DATA_PATH`
- `TARGET_COLUMN`
- `TEST_SIZE`
- `RANDOM_STATE`
- `MODEL_OUTPUT_PATH`
- `METRICS_OUTPUT_PATH`
- `CV_FOLDS`
- `N_ITER`

## Run Training

```bash
python -m src.pipelines.training_pipeline
```

Artifacts generated:

- `models/best_model.joblib`
- `artifacts/metrics.json`
- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/logs/churn_app.log`

## Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Endpoint

`POST /predict`

### Request

```json
{
  "tenure": 10,
  "MonthlyCharges": 70,
  "TotalCharges": 700,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check"
}
```

### Response

```json
{
  "predictions": [
    {
      "churn": "Yes",
      "probability": 0.82
    }
  ]
}
```

## Testing

```bash
pytest -q
```
