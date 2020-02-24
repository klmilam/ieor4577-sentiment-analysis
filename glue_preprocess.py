import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from preprocessing import preprocess

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "ieor_glue_data", table_name = "traintrain_csv", transformation_ctx = "datasource0")

applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("date", "string", "date", "string"), ("querytype", "string", "querytype", "string"), ("userid", "string", "userid", "string"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")

embedding = preprocess.PreprocessTweets(None, token_indices_json="artifacts.zip/token_indices.json").load_embedding_dictionary()

def run_pipeline(element):
    tweet = element["tweet"]
    features = preprocess.run_pipeline(tweet, 100, embedding)
    element["features"] = features
    return element

data = Map.apply(frame = applymapping1, f = run_pipeline)

datasink2 = glueContext.write_dynamic_frame.from_options(frame = data, connection_type = "s3", connection_options = {"path": "s3://ieore4577-klm2190/twitter/train"}, format = "json", transformation_ctx = "datasink2")
job.commit()