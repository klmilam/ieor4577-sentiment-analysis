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
