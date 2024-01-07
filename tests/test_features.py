
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer, OutlierHandler


def test_weathersit_variable_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config.weathersit_var,  
    )
    print("testing weathersit variable.....")
    print(sample_input_data.shape)
    print(sample_input_data.loc[12230,'weathersit'])
    assert np.isnan(sample_input_data.loc[12230,'weathersit'])

    print("going to fit and transform....")
    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    print("final assert...")
    # Then
    assert subject.loc[12230,'weathersit'] == 'Clear'
    
    
def test_temp_variable_outliers(sample_input_data):
    # Given
    outliers = OutlierHandler(
        variables=config.model_config.temp_var,
    )
    print('Temp min: ', sample_input_data['temp'].min())
    print('Temp max: ',sample_input_data['temp'].max())
    
    
    print('value at 12230: ', sample_input_data.loc[12230,'temp'])
    
    df = sample_input_data.copy()
    q1 = df.describe()['temp'].loc['25%']
    q3 = df.describe()['temp'].loc['75%']
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # update value to lower
    sample_input_data.loc[12230,'temp'] = lower_bound - 10
    
    print("going to fit and transform....")
    
    assert  sample_input_data.loc[12230,'temp'] > upper_bound or  sample_input_data.loc[12230,'temp'] < lower_bound 
    
    # When
    subject = outliers.fit(X=sample_input_data).transform(sample_input_data)
    print("final assert...")
    # Then
    assert subject.loc[12230,'temp'] == upper_bound or subject.loc[12230,'temp'] == lower_bound