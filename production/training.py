"""Processors for the model training step of the worklow."""
import logging
import os.path as op

from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    create_context,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    # features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # # sample data if needed. Useful for debugging/profiling purposes.
    # sample_frac = 0.5  # params.get("sampling_fraction", None)
    # if sample_frac is not None:
    #     logger.warn(f"The data has been sample by fraction: {sample_frac}")
    #     sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    # else:
    #     sample_X = train_X
    # sample_y = train_y.loc[sample_X.index]

    train_X = train_X[curated_columns]

    reg_ppln_ols = Pipeline([
        ('estimator', SKLStatsmodelOLS())
    ])
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # # transform the training data
    # train_X = get_dataframe(
    #     features_transformer.fit_transform(train_X, train_y),
    #     get_feature_names_from_column_transformer(features_transformer),
    # )
    # train_X = train_X[curated_columns]

    # # create training pipeline
    # reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # # fit the training pipeline
    # reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )


config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
param = {}
train_model(context, param)