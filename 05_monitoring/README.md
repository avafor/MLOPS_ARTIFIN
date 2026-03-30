# Iris Monitoring Pipeline

This project demonstrates a simple machine learning monitoring pipeline using the Iris dataset.

We simulate incoming data, compute monitoring metrics, store them in a database, and visualize them in Grafana.

---

## Pipeline Overview

1. Reference dataset (baseline)
2. Incoming batch data
3. Monitoring (drift + performance)
4. Storage in PostgreSQL
5. Visualization in Grafana

---

## Project Structure
```
.
├── data/
│   ├── reference.csv
│   └── current_batches/
├── scripts/
│   ├── prepare_reference.py
│   ├── generate_batch.py
│   └── calculate_metrics.py
├── docker-compose.yml
├── pyproject.toml
└── scaler.joblib
```
---

## How to Run

### 1. Install dependencies

poetry install

### 2. Start Docker

docker compose up -d

### 3. Set environment variables
```bash
export MLFLOW_TRACKING_URI="your_mlflow_tracking_uri"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=test
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=example
```
### 4. Run pipeline

```bash
poetry run python scripts/prepare_reference.py
poetry run python scripts/generate_batch.py
poetry run python scripts/calculate_metrics.py
```

Repeat last two steps multiple times.

---

## Adminer

http://localhost:8080

Login:
- System: PostgreSQL
- Server: db
- User: postgres
- Password: example
- Database: test

Query:
SELECT * FROM metrics;

---

## Grafana

http://localhost:3000

Login:
- admin / admin

Add PostgreSQL datasource:
- Host: db:5432
- Database: test
- User: postgres
- Password: example

---

## Example Queries

Accuracy:
```
SELECT timestamp AS time, accuracy FROM metrics;
```
Drift:
```
SELECT timestamp AS time, share_drifted_features FROM metrics;
```
---

## Diagram

![Monitoring Pipeline](https://github.com/user-attachments/assets/58338d0a-a3db-4b76-a2f3-57c1b5b832f9)

