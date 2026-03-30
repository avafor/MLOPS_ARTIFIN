# Iris Monitoring Pipeline

This project shows a simple machine learning monitoring pipeline using the Iris dataset, PostgreSQL, Adminer, and Grafana.

The goal is to simulate incoming data, calculate monitoring metrics, store them in a database, and visualize them in Grafana.

---

## Pipeline Overview

The pipeline works in four steps:

1. Create a **reference dataset** (`reference.csv`) that represents the healthy baseline.
2. Generate a **new batch of data** with predictions.
3. Compare the new batch with the reference data and compute monitoring metrics.
4. Store the metrics in PostgreSQL and visualize them in Grafana.

---

## Project Structure

```text
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

![Monitoring Pipeline](https://github.com/user-attachments/assets/58338d0a-a3db-4b76-a2f3-57c1b5b832f9)
