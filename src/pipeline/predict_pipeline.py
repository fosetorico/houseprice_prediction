import sys
import pandas as pd
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline: 
    def __init__(self):
        pass

    # model prediction pipe
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            print("Features before preprocessing:")
            print(features)            
            data_scaled=preprocessor.transform(features)
            print("Features after scaling:")
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        longitude: float,
        latitude: float,
        housing_median_age:int,
        total_rooms: int,
        total_bedrooms: int,
        population: int,
        households: int,
        median_income:float,
        ocean_proximity
        ):

        self.longitude = longitude
        self.latitude = latitude
        self.housing_median_age = housing_median_age
        self.total_rooms = total_rooms
        self.total_bedrooms = total_bedrooms
        self.population = population
        self.households = households       
        self.median_income = median_income       
        self.ocean_proximity = ocean_proximity       

    # return all input in form of a dataframe
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "longitude": [self.longitude],
                "latitude": [self.latitude ],
                "housing_median_age": [self.housing_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms ],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) 