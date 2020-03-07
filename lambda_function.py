import json
from preprocessing import preprocess

def lambda_handler(event, context):
    
    tweet = event["tweet"]
    zip_filename = "preprocessing/n artifacts.zip/token_indices.json"
    embedding = preprocess.PreprocessTweets(
        None,
        token_indices_json=zip_filename).load_embedding_dictionary()
    features = preprocess.run_pipeline(tweet, 40, embedding, 10000)

    return {
        'features': features
    }
