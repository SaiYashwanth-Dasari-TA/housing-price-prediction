"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op
import mlflow

from ta_lib.core.api import (load_dataset, load_pipeline, register_processor, save_dataset, create_context, DEFAULT_ARTIFACTS_PATH)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from mlflow import log_metric


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    # features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    # transform the test dataset
    # test_X = get_dataframe(
    #     features_transformer.transform(test_X),
    #     get_feature_names_from_column_transformer(features_transformer),
    # )
    test_X = test_X[curated_columns]

    # make a prediction
    test_X["yhat"] = model_pipeline.predict(test_X)
    # print("x", test_y.shape)
    # print("y", test_X["yhat"].shape)
    r_squared = r2_score(test_y, test_X["yhat"])
    mse = mean_squared_error(test_y, test_X["yhat"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, test_X["yhat"])
    # mpe = mean_percentage_error(test_X, test_X["yhat"])
    mape = mean_absolute_percentage_error(test_y, test_X["yhat"])
    # mase = mean_absolute_scaled_error(test_X, test_X["yhat"])

    # Log the metrics to MLflow
    log_metric("R-squared", r_squared)
    log_metric("MSE", mse)
    log_metric("RMSE", rmse)
    log_metric("MAE", mae)
    # log_metric("MPE", mpe)
    # log_metric("MASE", mase)
    log_metric("MAPE", mape)
    # store the predictions for any further processing.
    save_dataset(context, test_X, output_ds)


config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
param = {}
experiment_id = mlflow.create_experiment("experiment4")
with mlflow.start_run(
    run_name="PARENT_RUN",
    experiment_id=experiment_id,
):
    score_model(context, param)