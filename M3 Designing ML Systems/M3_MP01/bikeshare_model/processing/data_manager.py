import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from bikeshare_model.processing.features import WeathersitImputer

##  Pre-Pipeline Preparation

# 1. Extracts the year and month from the dteday
def get_year_and_month(dataframe :pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df[config.model_config.dteday_var] = pd.to_datetime(df[config.model_config.dteday_var], format='%Y-%m-%d')
    # print(df[config.model_config.dteday_var])
    # Add new features 'yr' and 'mnth
    df[config.model_config.yr_var] = df[config.model_config.dteday_var].dt.year
    df[config.model_config.mnth_var] = df[config.model_config.dteday_var].dt.month_name()
    return df

def weekday_imputer(df: pd.DataFrame) -> pd.DataFrame:
    # df = X.copy()
    wkday_null_idx = df[df[config.model_config.weekday_var].isnull() == True].index
    df.loc[wkday_null_idx, config.model_config.weekday_var] = df.loc[wkday_null_idx, config.model_config.dteday_var].dt.day_name().apply(lambda x: x[:3])
    # print("Weekday imputation is done...")
    return df

def weekday_oneHotEncoder(X: pd.DataFrame):
    encoder_ = OneHotEncoder(sparse_output=False)
    
    encoded_weekday = encoder_.fit(X[['weekday']]).transform(X[['weekday']])
    # Get encoded feature names
    enc_wkday_features = encoder_.get_feature_names_out(['weekday'])
    X[enc_wkday_features] = encoded_weekday
    # print(X.head())
    return X

# 2. processing 

f1=lambda x: 0 if type(x) == float else 1

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame = get_year_and_month(data_frame)      # Extracts the year and month
    data_frame = weekday_imputer(data_frame) # Weekday imputation
    data_frame = weekday_oneHotEncoder(data_frame) #one hot encoding for weekday
    
    weatherImputer = WeathersitImputer(variables=config.model_config.weathersit_var)
    data_frame = weatherImputer.fit(data_frame).transform(data_frame)
    # drop unnecessary variables
    cols_to_drop =[col for col in config.model_config.unused_fields if col in data_frame.columns]
    data_frame.drop(labels=cols_to_drop, axis=1, inplace=True)
     
    return data_frame

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
            
def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model