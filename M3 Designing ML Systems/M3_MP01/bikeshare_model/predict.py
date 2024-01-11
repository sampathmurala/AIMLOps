import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd 

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    validated_data=validated_data.reindex(columns=config.model_config.features)
    # print(validated_data)
    
    predictions = bikeshare_pipe.predict(validated_data)
    results = {"predictions": predictions,"version": _version, "errors": errors}
    # print(results) 
    
    return results

if __name__ == "__main__":
    data_in={ "season":['summer'],"hr":['8pm'],'holiday':['No'], 'workingday':['No'],'weathersit':['Light Rain'],'temp':[8.92], 
             'atemp':[5.9978], 'hum':[93.0],'windspeed':[27.9993],  'weekday':['Sun'], 'dteday':['2012-04-22']}
    # data_in['dteday'] = pd.to_datetime(data_in['dteday']).strftime('%Y-%m-%d')
    
    result = make_prediction(input_data=data_in)
    print(result)
