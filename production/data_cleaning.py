"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
# import mlflow
# import os.path as op
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    load_dataset,
    register_processor,
    save_dataset,
    # create_context
)


@register_processor("data-cleaning", "housing")
def clean_product_table(context, params):

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_clean = (
        housing_df

        .copy()

        .passthrough()

        .replace({'': np.NaN})

        .clean_names(case_type='snake')

    )
    housing_df_clean.dropna(subset=['total_bedrooms'], inplace=True)

    housing_df_clean.reset_index(drop=True, inplace=True)
    # housing_df_clean.head()

    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_clean = load_dataset(context, input_dataset)

    housing_df_clean["income_cat"] = pd.cut(
        housing_df_clean["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_df_clean, housing_df_clean["income_cat"]):
        strat_train_set = housing_df_clean.loc[train_index]
        strat_test_set = housing_df_clean.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing_tr_features = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_tr_target = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_tr_f_num = housing_tr_features.drop("ocean_proximity", axis=1)

    imputer.fit(housing_tr_f_num)
    X = imputer.transform(housing_tr_f_num)

    housing_tr_fe = pd.DataFrame(X, columns=housing_tr_f_num.columns, index=housing_tr_features.index)
    housing_tr_fe["rooms_per_household"] = housing_tr_fe["total_rooms"] / housing_tr_fe["households"]
    housing_tr_fe["bedrooms_per_room"] = (
        housing_tr_fe["total_bedrooms"] / housing_tr_fe["total_rooms"]
    )
    housing_tr_fe["population_per_household"] = (
        housing_tr_fe["population"] / housing_tr_fe["households"]
    )

    housing_tr_cat = housing_tr_features[["ocean_proximity"]]
    housing_tr_features = housing_tr_fe.join(pd.get_dummies(housing_tr_cat, drop_first=True))

    housing_te_features = strat_test_set.drop("median_house_value", axis=1)
    housing_te_target = strat_test_set["median_house_value"].copy()

    housing_te_f_num = housing_te_features.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(housing_te_f_num)
    housing_te_fe = pd.DataFrame(
        X_test_prepared, columns=housing_te_f_num.columns, index=housing_te_features.index
    )
    housing_te_fe["rooms_per_household"] = (
        housing_te_fe["total_rooms"] / housing_te_fe["households"]
    )
    housing_te_fe["bedrooms_per_room"] = (
        housing_te_fe["total_bedrooms"] / housing_te_fe["total_rooms"]
    )
    housing_te_fe["population_per_household"] = (
        housing_te_fe["population"] / housing_te_fe["households"]
    )

    housing_te_cat = housing_te_features[["ocean_proximity"]]
    housing_te_features = housing_te_fe.join(pd.get_dummies(housing_te_cat, drop_first=True))
    housing_te_features['ocean_proximity_ISLAND'] = 0

    save_dataset(context, housing_tr_features, output_train_features)
    save_dataset(context, housing_tr_target, output_train_target)

    save_dataset(context, housing_te_features, output_test_features)
    save_dataset(context, housing_te_target, output_test_target)


# config_path = op.join('conf', 'config.yml')
# context = create_context(config_path)
# param = {}
# experiment_id = mlflow.create_experiment("experiment5")
# with mlflow.start_run(
#     run_name="PARENT_RUN",
#     experiment_id=experiment_id,
# ):
#     clean_product_table(context, param)
