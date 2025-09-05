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
