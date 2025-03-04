import joblib
import numpy as np
import pandas as pd 
from pathlib import Path 

class PredictionPipeline:
    def __init__(self):
        self.model=joblib.load("artifacts/model_trainer/model_joblib")

    def predict(self,data):
        prediction=self.model.predict(data)

        return prediction    
