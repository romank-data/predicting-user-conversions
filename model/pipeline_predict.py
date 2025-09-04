import pandas as pd
import datetime
import dill
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Пути к файлам
ENCODER_PATH = "model/encoder.pkl"
SCALER_PATH = "model/scaler.pkl"
MODEL_PATH = "model/model_sber_auto.pkl"
JSON_PATH = "model/check.json"
PREDICTIONS_PATH = "/model/predictions.csv"

def load_data(filepath):
    df = pd.read_parquet(filepath)
    X = df.drop(columns=['is_target_action'])
    y = df['is_target_action']
    return X, y

def train_model(X, y):
    parameters_rf = {'min_samples_split': [2, 3, 4]}
    rf = RandomForestClassifier(
        n_estimators=100, max_features='sqrt', min_samples_leaf=2,
        bootstrap=False, max_depth=100, n_jobs=-1, random_state=42
    )
    clf_rf = GridSearchCV(rf, parameters_rf, cv=4, scoring='roc_auc', verbose=100)
    clf_rf.fit(X, y)
    score = roc_auc_score(y, clf_rf.predict_proba(X)[:, 1])
    return clf_rf, score

def save_model(model, score, filename=MODEL_PATH):
    with open(filename, 'wb') as file:
        dill.dump({
            'model': model,
            'metadata': {
                'name': 'sber target model',
                'author': 'Roman Kostenko',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(model).__name__,
                'ROC_AUC': score
            }
        }, file)
    print(f"{filename}, ROC_AUC: {score:.4f}")

def load_model_and_transformers():
    with open(MODEL_PATH, "rb") as f:
        model_data = dill.load(f)
    model = model_data["model"]
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, ohe, scaler

# Преобразование данных
def transform_data(data, ohe):
    categorical_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                            'utm_keyword', 'device_category', 'device_os', 'device_brand', 'device_browser']
    numerical_features = ['hour', 'dayofweek', 'device_width', 'device_height']
    binary_features = ['is_weekend','is_organic', 'from_social', 'is_russia', 'is_moscow_spb']

    # Заполняем отсутствующие бинарные признаки
    for col in binary_features:
        data[col] = data.get(col, 0)

    # Заполняем отсутствующие числовые признаки
    for col in numerical_features:
        if col not in data.columns:
            data[col] = 0

    # One-Hot Encoding
    ohe_df = pd.DataFrame(ohe.transform(data[categorical_features]),columns=ohe.get_feature_names_out(categorical_features))

    # Объединяем бинарные, числовые и OHE-признаки
    data = pd.concat([data[binary_features], ohe_df], axis=1)

    return data

# Загрузка данных из JSON
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # Преобразуем в список, если это одиночный объект
        data = [data]
    return pd.DataFrame(data)

# Получение предсказаний
def predict_on_data(model, data, threshold=0.05):
    predictions = model.predict_proba(data)[:, 1]
    binary_predictions = (predictions > threshold).astype(int)

    print(f"1 in preds: {binary_predictions.sum()} из {len(binary_predictions)}")

    return pd.DataFrame(binary_predictions, columns=["is_target_action"])


# Основной блок
if __name__ == "__main__":
    # Шаг 1: Обучение модели
    data_path = "model/df_prepared.parquet"
    X, y = load_data(data_path)
    model, score = train_model(X, y)
    save_model(model, score)

    df_new_data = load_json(JSON_PATH)
    model, ohe, scaler = load_model_and_transformers()

    df_transformed = transform_data(df_new_data, ohe)

    predictions = predict_on_data(model, df_transformed)

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print("Preds are saved as predictions.csv")
