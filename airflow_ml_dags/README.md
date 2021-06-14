# Домашняя работа №3
Соколов Александр
## Установка 
``` 
git clone https://github.com/made-ml-in-prod-2021/sokoloid
cd sokoloid
git checkout homework3
cd airflow_ml_dags
```
## запуск
для отправки повещений по электронной почте необходимо 
указать данные учетной записи и разрешить в соотвествующем  аккануте подключение с ногово устройства
```
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export MAIL_USER=aesokolov1975@gmail.com
export MAIL_PASSWD=*********
sudo -E docker-compose up --build
```
## Фотографии работы 

 1
 ![2](https://user-images.githubusercontent.com/46603429/121949114-c6c0a300-cd60-11eb-82ac-6e0c872dd578.png)
2![3](https://user-images.githubusercontent.com/46603429/121949133-cb855700-cd60-11eb-9f15-a2748323a97c.png)
3![5](https://user-images.githubusercontent.com/46603429/121949152-d344fb80-cd60-11eb-9efb-cb410c2f3a88.png)
4![6](https://user-images.githubusercontent.com/46603429/121949165-d63fec00-cd60-11eb-9772-b83947bac998.png)
5![7](https://user-images.githubusercontent.com/46603429/121949209-dd66fa00-cd60-11eb-8080-708891736b4c.png)


## Самооценка
В ДЗ предлагается на основе airflow реализовать описанную выше схему, к деталям:

 
1) (5 баллов) Реализуйте dag, который генерирует данные для обучения модели )
2) (10 баллов) Реализуйте dag, который обучает модель еженедельно,
3) (5 баллов) Реализуйте dag, который использует модель ежедневно 
3а) (3 доп балла)  Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения
4) (10 баллов) все даги реализованы только с помощью DockerOperator  /blob/main/dags/11_docker.py#L27 в этом месте пробрасывается путь с хостовой машины, используйте здесь путь типа /tmp или считывайте из переменных окружения.

5) (0) Протестируйте ваши даги ... 
6) (0) В docker compose так же настройте поднятие mlflo...
7)(0) вместо пути в airflow variables  используйте апи Mlflow
8) (3 доп. балла) Настройте alert в случае падения дага 
9) (1 балл) традиционно, самооценка 

5+10+5+3+10+0+0+0+3+1 = 37
Дисконт по времени
37 * 0.6 =22.2  (Разборки со старым процессором и его последущая замена заняли время)
