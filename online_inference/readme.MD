PET проект по определению типа территории по GPS координатам и изображению генплана


Сделано:
0) ветка   homework2,   код в папке online_inference

1)  inference   модели обернут в rest сервис FastAPI: online_inference/scr/model/app_fast_api.py .(3 балла)

2) Сделан  тест (pytest) для /predict:    online_inference/test/test_web.py (3 балла)

3) Написан  скрипт, который будет делает запросы к   сервису  online_inference/test/script_test_web.py-- (2 балла)

4) Сделана валидация входных данных (проверка формата и значений GPS координат. при неверной валидации код возврата 400) - (3 доп балла)

5) Написан Dockerfile . На его основе собран докер-контейнер. (4 балла) 
   - Сборка  docker build -t sokoloid/geo_map:v1 .
   - Запусак docker run -p 8000:8000 sokoloid/geo_map:v1 
   


6) Для оптимизации докер-контейнера исаользован образ python:3.8.10-slim  (3 балла)

7) Докер образ опубликован на dockerhub   sokoloid/geo_map:v1    (2балла)

8) Команды для локального запуска контейнера на порту 8000 
     - docker pull  sokoloid/geo_map:v1
     - docker run -p 8000:8000 sokoloid/geo_map:v1 

9) самооценка -- 1 доп балл

Итого: 3+3+2+3+4+3+2+1= 21 балл.