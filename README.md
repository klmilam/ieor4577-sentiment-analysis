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

## Glue ETL
The Glue ETL script is available at `glue_preprocess.py`

## Run training
```bash
export SM_CHANNEL_EVAL="s3://ieore4577-klm2190/twitter/eval"
export SM_CHANNEL_VALIDATION="s3://ieore4577-klm2190/twitter/dev"
export SM_CHANNEL_TRAIN="s3://ieore4577-klm2190/twitter/train"
export S3_REQUEST_TIMEOUT_MSEC=600000
python3 -m model_training.sentiment_training 2>&1 | grep -v "Connection has been released. Continuing."
```

Screenshot of results can be found at `Training_results.png`

## Run Training on AI Platform
### Train locally
```bash
gcloud ai-platform local train \
  --package-path gcp_model_training \
  --module-name gcp_model_training.sentiment_training \
  --job-dir gs://internal-klm/sentiment-analysis/model/sentiment_analysis_$(date +%Y%m%d%H%M%S)
```

### Train on AI Platform
```bash
gcloud ai-platform jobs submit training sentiment_analysis_$(date +%Y%m%d%H%M%S) \
  --package-path gcp_model_training \
  --module-name gcp_model_training.sentiment_training \
  --job-dir gs://internal-klm/sentiment-analysis/model/sentiment_analysis_$(date +%Y%m%d%H%M%S) \
  --region us-central1 \
  --python-version 3.5 \
  --runtime-version 1.14 \
  --stream-logs
```

### Hyperparameter Tuning
```bash
gcloud ai-platform jobs submit training sentiment_analysis_$(date +%Y%m%d%H%M%S) \
  --package-path gcp_model_training \
  --config gcp_model_training/hptuning.yaml \
  --module-name gcp_model_training.sentiment_training \
  --job-dir gs://internal-klm/sentiment-analysis/model/sentiment_analysis_$(date +%Y%m%d%H%M%S) \
  --region us-central1 \
  --python-version 3.5 \
  --runtime-version 1.14 \
  --stream-logs
```
