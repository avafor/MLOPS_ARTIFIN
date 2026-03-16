from train_predict_scheduled import iris_monthly_train_and_batch_predict

if __name__ == "__main__":
    iris_monthly_train_and_batch_predict.serve(
        name="iris-monthly",
        cron="35 14 16 * *",   
        parameters={"n_samples": 100},
    )