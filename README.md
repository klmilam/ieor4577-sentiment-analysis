# Sentiment Analysis for Tweets


## Set Up Python Environment
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Run pylint and pytest for Preprocessing
```bash
pytest --cov=preprocessing
pylint preprocessing/test_preprocess.py
pylint pylint preprocessing/preprocess.py
```

## Run training
```bash
export SM_CHANNEL_EVAL=data/eval
export S3_REQUEST_TIMEOUT_MSEC=600000
python3 -m model_training.sentiment_training 2>&1 | grep -v "Connection has been released. Continuing."
```