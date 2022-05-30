online inference
==============================

ml in prod homework 2

# ML production ready project example
## How to use

unit test:
```commandline
cd online_inference
python -m pytest test/predict_test.py
```

###Inference scripts:

Healt:
```commandline
curl -X 'GET' \
  'http://0.0.0.0:PORT/health' \
  -H 'accept: application/json'
```
Predict:
```commandline
curl -X 'GET' \
  'http://0.0.0.0:PORT/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data":[
    [69.0, 1.0, 0.0, 160.0, 234.0, 1.0, 2.0, 131.0, 0.1, 1.0, 1.0, 1.0, 0.0]
  ],
  "feature_names": [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
  ]
}'
```

--------