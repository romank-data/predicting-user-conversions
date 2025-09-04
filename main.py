import json

import dill
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

ENCODER_PATH = "/Users/romankostenko/PycharmProjects/sber_auto/model/data/encoder.pkl"
SCALER_PATH = "/Users/romankostenko/PycharmProjects/sber_auto/model/data/scaler.pkl"
MODEL_PATH = "/Users/romankostenko/PycharmProjects/sber_auto/model/data/model_sber_auto.pkl"

app = FastAPI()

with open('/Users/romankostenko/PycharmProjects/sber_auto/model/data/model_sber_auto.pkl', 'rb') as f:
    model = dill.load(f)
ohe = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

class Form(BaseModel):
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_browser: object
    is_target_action: int
    hour: int
    dayofweek: int
    is_weekend: int
    device_width: int
    device_height: int
    is_organic: int
    from_social: int
    is_russia: int
    is_moscow_spb: int


class Prediction(BaseModel):
    pred: int

@app.get('/status')
def status():
    return {"I'm OK"}

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    # Преобразование данных (аналогично pipeline коду):
    categorical_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                            'utm_keyword', 'device_category', 'device_os', 'device_brand', 'device_browser']
    numerical_features = ['hour', 'dayofweek', 'device_width', 'device_height']
    binary_features = ['is_weekend', 'is_organic', 'from_social', 'is_russia', 'is_moscow_spb']

    # Заполняем отсутствующие бинарные признаки
    for col in binary_features:
        df[col] = df.get(col, 0)

    # Заполняем отсутствующие числовые признаки
    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0

    ohe_df = pd.DataFrame(ohe.transform(df[categorical_features]),
                          columns=ohe.get_feature_names_out(categorical_features))

    df = pd.concat([df[binary_features], ohe_df], axis=1)

    expected_features = model['model'].feature_names_in_  # Фичи, которые использовались при обучении
    missing_features = set(expected_features) - set(df.columns)
    extra_features = set(df.columns) - set(expected_features)

    # Добавляем отсутствующие фичи (заполняем нулями)
    for feature in missing_features:
        df[feature] = 0

    # Удаляем лишние фичи
    df = df[expected_features]

    y = model['model'].predict(df)

    probabilities = model['model'].predict_proba(df)[:, 1]
    y[0] = (probabilities > 0.05).astype(int)[0]
    return {#"id": form.id,
            "pred": y[0],
            #"price": form.price
            }


