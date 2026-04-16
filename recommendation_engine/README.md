# Recommendation Engine

Production-style recommendation system for movies/products using collaborative filtering, content-based ranking, and a hybrid recommender.

## Business Problem

Suggest items to users based on historical behavior, similar users, and item similarity. The system is designed like a real production recommender: it supports cold start fallback, offline training, persisted artifacts, and a live API.

## Architecture

```text
recommendation_engine/
  data/
    raw/
    processed/
  notebooks/
    eda.ipynb
  src/
    components/
      data_loader.py
      preprocessing.py
      collaborative_filtering.py
      content_based.py
      hybrid_model.py
      evaluation.py
    pipelines/
      training_pipeline.py
      recommendation_pipeline.py
    utils/
      io_utils.py
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

1. Data ingestion for user-item interactions and item metadata.
2. Data validation for schema consistency, null keys, and signal source checks.
3. Preprocessing that blends explicit ratings with behavioral signals.
4. Per-user chronological split for realistic offline evaluation.
5. Collaborative filtering using user-user and item-item similarity.
6. Content-based ranking using TF-IDF item embeddings.
7. Hybrid model combining collaborative, content, and popularity signals.
8. Offline hybrid weight tuning on a validation split.
9. Evaluation with Precision@K, Recall@K, and RMSE.
10. Cold start handling via popularity-based fallback and top-N controls.

## Production Enhancements

- Config-driven model and pipeline controls (`TOP_K_EVAL`, `MAX_TOP_N`, etc.).
- Structured logging with API request timing and inference traces.
- Inference caching in recommendation pipeline for repeated user requests.
- Robust API validation via Pydantic constraints and response schema.
- Expanded tests for cold start, seen-item filtering, and API behavior.
- Persisted artifacts include metrics, tuned weights, and metadata.

### Environment Variables

- `INTERACTIONS_PATH`
- `ITEMS_PATH`
- `MODEL_PATH`
- `METRICS_PATH`
- `TOP_K_NEIGHBORS`
- `TOP_K_EVAL`
- `DEFAULT_TOP_N`
- `MAX_TOP_N`
- `CF_WEIGHT`
- `CB_WEIGHT`
- `POPULARITY_WEIGHT`

## Data Format

Place these files in `data/raw/`:

- `interactions.csv`
- `items.csv`

Expected interactions columns:

- `user_id`
- `item_id`
- `rating`
- `interaction_type`
- `timestamp`

Expected items columns:

- `item_id`
- `title`
- `genres`
- `description`

Sample data is included so the project runs out of the box.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.pipelines.training_pipeline
```

Artifacts generated:

- `models/hybrid_recommender.joblib`
- `artifacts/metrics.json`
- `data/processed/interactions_processed.csv`
- `data/processed/items_processed.csv`

## Serve API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Endpoint

`POST /recommend`

### Request

```json
{
  "user_id": 101,
  "top_n": 3
}
```

### Response

```json
{
  "recommendations": ["i6", "i9", "i2"]
}
```

### Health Check

`GET /health`

## Notebook

Open `notebooks/eda.ipynb` for:

- User activity distribution
- Popular item analysis
- Sparsity analysis

## Test

```bash
pytest -q
```
