import pandas as pd
import numpy as np
import joblib

def load_model_components():
    model = joblib.load('model.joblib')
    poly_features = joblib.load('poly_features.joblib')
    q = joblib.load('q.joblib')
    mapping = joblib.load('mapping.joblib')
    reverse_mapping = joblib.load('reverse_mapping.joblib')
    return model, poly_features, q, mapping, reverse_mapping

def estimate_y(model, poly_features, q, vd, hd, cont):
    input_data = pd.DataFrame({'VD': [vd], 'HD': [hd], 'CONT_': [cont]})
    input_poly = poly_features.transform(input_data)
    y_pred = model.predict(input_poly)[0]
    return y_pred, y_pred - q, y_pred + q