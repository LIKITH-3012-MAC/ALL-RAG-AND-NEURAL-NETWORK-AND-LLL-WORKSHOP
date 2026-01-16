import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import yaml
import pickle

# -------------------- LOAD CONFIG --------------------
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

X = np.array(config["data"]["x"])
y = np.array(config["data"]["y"])
predict_input = np.array([config["data"]["predict"]])

# -------------------- MLflow SETUP --------------------
mlflow.set_experiment("Linear_Regression_YAML_Demo")

with mlflow.start_run():

    # -------------------- MODEL --------------------
    model = LinearRegression()
    model.fit(X, y)

    # -------------------- PREDICTION --------------------
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    prediction = model.predict(predict_input)

    # -------------------- LOG PARAMETERS --------------------
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("num_samples", len(X))
    mlflow.log_param("input_feature_dim", X.shape[1])

    # -------------------- LOG METRICS --------------------
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # -------------------- SAVE MODEL AS PKL --------------------
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # -------------------- LOG ARTIFACT --------------------
    mlflow.log_artifact("linear_model.pkl")

    # -------------------- LOG MODEL TO MLflow --------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="LinearRegression_YAML_Model"
    )

    # -------------------- OUTPUT --------------------
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    print(f"Prediction for input {predict_input[0][0]}: {prediction[0]}")
