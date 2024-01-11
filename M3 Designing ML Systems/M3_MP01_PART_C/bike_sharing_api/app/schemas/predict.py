from typing import Any, List, Optional

from pydantic import BaseModel
from bikeshare_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "season": "summer",
                        "hr": '8pm',
                        "holiday": "No",
                        "workingday": "No",
                        "weathersit": "Light Rain",
                        "temp": 8.92,
                        "atemp": 5.9978,
                        "hum": 93.0,
                        "windspeed": 27.9993,
                        "yr": 2012,
                        "mnth": 'April',
                        "weekday": 'Sun',
                        "dteday": '2012-04-22T00:00:00',
                    }
                ]
            }
        }
