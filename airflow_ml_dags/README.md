# airflow-hw3 ZibyLC
код для пары Data Pipelines

чтобы развернуть airflow, предварительно собрав контейнеры

Старт вебсервер, бд, воркер
```console
cd airflow_ml_dags
docker compose up --force-recreate --build -d
```

Ссылка на документацию по docker compose up

https://docs.docker.com/compose/reference/up/