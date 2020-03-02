from datetime import datetime
import json
import boto3

from preprocessing import preprocess

sagemaker_client = boto3.client("runtime.sagemaker")
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    request_time = datetime.now()
    timestamp = request_time.strftime('%Y%m%d%H%M%S')

    tweet = event["tweet"]
    zip_filename = "preprocessing/artifacts.zip/token_indices.json"
    embedding = preprocess.PreprocessTweets(
        None,
        token_indices_json=zip_filename).load_embedding_dictionary()
    features = preprocess.run_pipeline(tweet, 40, embedding, 10000)

    preprocess_time = datetime.now()

    model_payload = {
        'features_input': features
    }

    model_response = sagemaker_client.invoke_endpoint(
        EndpointName="sentiment-inference-endpoint",
        ContentType="application/json",
        Body=json.dumps(model_payload))

    prediction = json.loads(model_response["Body"].read().decode())

    prediction_time = datetime.now()

    response = {}

    if prediction["predictions"][0][0] >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"

    payload_log = {}
    payload_log["sentiment"] = response["sentiment"]
    payload_log["prediction"] = prediction["predictions"][0][0]
    payload_log["tweet"] = tweet
    payload_log["request_time"] = str(request_time)
    payload_log["preprocess_time"] = str(preprocess_time - request_time)
    payload_log["inference_time"] = str(prediction_time - preprocess_time)

    filename = "payloads/results-" + timestamp + ".json"
    s3_object = s3.Object("ieore4577-klm2190", filename)
    s3_object.put(Body=bytes(json.dumps(payload_log).encode("UTF-8")))

    return response
