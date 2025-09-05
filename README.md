ДОДЕЛАТЬ README !!!

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHpodWtuZWQxZW9heGtxcnhzMjdrdTN6YWp2b3lzbDh4bnZ2emE3aSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mIMsLsQTJzAn6/giphy.gif" width="100"/>

# 🚗 ML Project Sberauto

>  "Кто не умеет писать ML-модели, тот не сдал введение в специальность на тройку в первом семестре" © Кураторы высшего образования Skillbox

## 🚀 Как запустить модель

Модель написана на Fast API, поэтому для запуска необходимо в терминале запустить команду 

```sh
uvicorn main:app --reload
```

## 🗿 Тестирование модели

Для тестирования модели можно воспользоваться двумя вариантами:
- Готовым файлом для тестирования **conversion-score-git.ipynb**
- Перейти в Swagger (обычно это http://127.0.0.1:8000/docs на локальном компьютере) и протестировать с помощью вызова модуля predict

## 😋 Шаблон входных данных и результата работы модели

Передача входных данных:

```json
{
"utm_source": "dGlVSdmIlgWDyOPjfwwy",
"utm_medium": "partner",
"utm_campaign": "LTuZkdKfxRGVceoWkVyg",
"utm_adcontent": "JNHcPlZPxEMWDnRiyoBf",
"utm_keyword": "puhZPIYqKXeFPaUviSjo",
"device_category": "mobile",
"device_os": "Android",
"device_brand": "Samsung",
"device_browser": "YaBrowser",
"is_target_action": 0,
"hour": 0,
"dayofweek": 5,
"is_weekend": 1,
"device_width": 360,
"device_height": 800,
"is_organic": 0,
"from_social": 0,
"is_russia": 1,
"is_moscow_spb": 1
}
```

Результат работы модели:

```json 
{
  "predict": 1
}
```

🚀 Model for Predicting Conversion on 'SberAvtopodpiska' Website
📊 Project Overview
This project is a machine learning model development for predicting user conversion on the "SberAvtopodpiska" service website. The main goal is to help the product team identify the most effective traffic sources and user behavior that lead to conversion.

The model solves a real-world business problem by using Google Analytics data to predict whether a user will perform a target action, such as "Submit an application" or "Order a callback."

✨ Key Features
Data Preprocessing: The project includes the merging of two datasets (ga_sessions and ga_hits) and extensive data cleaning. Missing values were handled, duplicates were removed, and data types were converted.

Feature Engineering: New, more informative features were created, such as is_organic (organic traffic) and from_social (traffic from social networks), and time-based features (hour, dayofweek) were extracted from the source data.

Model Training and Evaluation: Several classification algorithms were used for prediction: Logistic Regression, Random Forest, and Multi-Layer Perceptron (MLP). The models were optimized using GridSearchCV, and ROC-AUC was chosen as the main quality metric, with a target value of around 0.65.

Model Serialization: The best model was saved using dill for future use and deployment.

API Implementation: The project includes a simple API that accepts user visit data and returns a prediction (0 or 1).

🛠️ Technologies Used
Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

dill

joblib

IPython

tqdm

🚀 Installation and Setup
To run the project locally, follow these steps:

Clone the repository:

git clone [https://github.com/romank-data/sber-conversion-score.git](https://github.com/romank-data/sber-conversion-score.git)
cd sber-conversion-score

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate # For Windows: `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Download the dataset:
The project uses Google Analytics data. You will need to download the ga_sessions.csv and ga_hits.csv files. These files were originally obtained from Kaggle, but for local execution, you will need to find and download them manually.

Run the Jupyter Notebook:

jupyter notebook conversion-score-github.ipynb

Open conversion-score-github.ipynb in your browser and run all cells to reproduce the data analysis and model training.

📈 Results and Conclusions
The data analysis revealed several key insights:

The Random Forest Classifier proved to be the best model, achieving a ROC-AUC of 0.7383 on the full dataset.

Users from organic traffic (with an acquisition type of organic, referral, (none)) showed a higher conversion rate (CR) to target events.

Traffic from social networks had a lower CR compared to other sources.

These findings provide valuable information for the "SberAvtopodpiska" team, allowing them to adjust their marketing strategy and focus on the most effective traffic acquisition channels.

🤝 Contribution
You are welcome to fork this repository, create issues, or submit pull requests. Any contribution is appreciated!

📧 Contacts
If you have any questions or suggestions, please contact:

Roman Kostenko — GitHub Profile

Email: roman.kostenko@hotmail.com
