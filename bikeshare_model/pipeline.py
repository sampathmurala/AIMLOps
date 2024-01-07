import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder

bikeshare_pipe=Pipeline([

    ('weathersit_imputation', WeathersitImputer(variables=config.model_config.weathersit_var)),
    ('weekday_imputation', WeekdayImputer(variables=config.model_config.weekday_var)),
    ##==========Mapper======##
    ('map_yr',Mapper(config.model_config.yr_var, config.model_config.yr_mapping)),
    ('map_mnth',Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping )),
    ('map_season',Mapper(config.model_config.season_var, config.model_config.season_mapping )),
    ('map_weathersit',Mapper(config.model_config.weathersit_var, config.model_config.weather_mapping )),
    ('map_holiday',Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping )),
    ('map_workingday',Mapper(config.model_config.workingday_var,config.model_config.workingday_mapping )),
    ('map_hr',Mapper(config.model_config.hr_var, config.model_config.hour_mapping )),
    #Outlier handlers
    ('outlier_temp', OutlierHandler(config.model_config.temp_var)),
    ('outlier_atemp', OutlierHandler(config.model_config.atemp_var)),
    ('outlier_hum', OutlierHandler(config.model_config.hum_var)),
    ('outlier_windspeed', OutlierHandler(config.model_config.windspeed_var)),
    #weekday OneHotEncoder
    ('Weekday_OneHot', WeekdayOneHotEncoder()),
    # scale
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth, random_state=config.model_config.random_state))
])