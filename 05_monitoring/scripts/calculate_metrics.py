import os
import glob
from datetime import datetime

import pandas as pd
import psycopg
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    # load reference data
    reference = pd.read_csv("data/reference.csv")

    # load latest batch
    batch_files = glob.glob("data/current_batches/*.csv")
    latest_file = max(batch_files, key=os.path.getmtime)
    current = pd.read_csv(latest_file)
    batch_id = os.path.basename(latest_file).replace(".csv", "")

    # compute simple drift metric
    drifted_features = 0
    for feature in FEATURES:
        _, p_value = ks_2samp(reference[feature], current[feature]) # KS test for distribution difference 
        if p_value < 0.05: # if p-value is less than 0.05, we consider the feature to be drifted
            drifted_features += 1

    share_drifted = drifted_features / len(FEATURES)

    # compute model performance
    accuracy = accuracy_score(current["target"], current["prediction"])
    f1_macro = f1_score(current["target"], current["prediction"], average="macro") # why macro? because we have 3 classes and want to give equal weight to each class regardless of their frequency

    # compute prediction shares
    pred_share = current["prediction"].value_counts(normalize=True).to_dict() # get the share of each predicted class in the batch
    pred_setosa = pred_share.get(0, 0.0) # get the share of class 0 (setosa), if not present, default to 0.0
    pred_versicolor = pred_share.get(1, 0.0)
    pred_virginica = pred_share.get(2, 0.0)

    # connect to Postgres
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
    )

    # create table if needed
    with conn.cursor() as cur: # use cursor to execute SQL commands
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP,
                batch_id TEXT,
                batch_size INT,
                num_drifted_features INT,
                share_drifted_features FLOAT,
                accuracy FLOAT,
                f1_macro FLOAT,
                pred_setosa_share FLOAT,
                pred_versicolor_share FLOAT,
                pred_virginica_share FLOAT
            );
        """)

        # insert one row
        cur.execute("""
            INSERT INTO metrics (
                timestamp,
                batch_id,
                batch_size,
                num_drifted_features,
                share_drifted_features,
                accuracy,
                f1_macro,
                pred_setosa_share,
                pred_versicolor_share,
                pred_virginica_share
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            batch_id,
            len(current),
            drifted_features,
            share_drifted,
            accuracy,
            f1_macro,
            pred_setosa,
            pred_versicolor,
            pred_virginica
        ))

    conn.commit()
    conn.close()

    print("metrics saved to database")


if __name__ == "__main__":
    main()