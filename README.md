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
- Готовым файлом для тестирования **Проверка работы модели.ipynb**
- Перейти в Swagger (обычно это http://127.0.0.1:8000/docs на локальном компьютере) и протестировать с помощью вызова модуля predict

## 😋 Шаблон входных данных и результата работы модели

Передача входных данных:

```json
{
  "utm_source": "fDLlAcSmythWSCVMvqvL",
  "utm_medium": "(none)",
  "utm_campaign": "LTuZkdKfxRGVceoWkVyg",
  "utm_adcontent": "JNHcPlZPxEMWDnRiyoBf",
  "utm_keyword": "puhZPIYqKXeFPaUviSjo",
  "device_category": "mobile",
  "device_os": "iOS",
  "device_brand": "Xiaomi",
  "device_model": "iPhone",
  "device_screen_resolution": "360x720",
  "device_browser": "Chrome",
  "geo_country": "Russia",
  "geo_city": "Moscow"
}
```

Результат работы модели:

```json 
{
  "predict": 1
}
```
