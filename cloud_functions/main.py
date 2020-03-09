import preprocess

import googleapiclient.discovery
from datetime import datetime

from flask import json

def predict(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_time = datetime.now()
    timestamp = request_time.strftime('%Y%m%d%H%M%S')

    request_json = request.get_json()
    tweet = request_json["tweet"]
    embedding = preprocess.PreprocessTweets(
        None,
        token_indices_json="gs://internal-klm/sentiment-analysis/token_indices.json").load_embedding_dictionary()
    features = preprocess.run_pipeline(tweet, 40, embedding, 10000)

    preprocess_time = datetime.now()
    service = googleapiclient.discovery.build('ml', 'v1', cache_discovery=False)
    name = 'projects/{}/models/{}'.format("internal-klm", "sentiment_analysis_tuned")
    response = service.projects().predict(
        name=name,
        body={'instances': {"embedding_input": features}}
    ).execute()
    prediction = response["predictions"]
    prediction_time = datetime.now()
    response = {}

    if prediction[0]["dense_3"][0] >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"

    payload_log = {}
    payload_log["sentiment"] = response["sentiment"]
    payload_log["prediction"] = prediction[0]["dense_3"][0]
    payload_log["tweet"] = tweet
    payload_log["request_time"] = str(request_time)
    payload_log["preprocess_time"] = str(preprocess_time - request_time)
    payload_log["inference_time"] = str(prediction_time - preprocess_time)
    return json.dumps(response)
