import json
import tensorflow as tf

class ConfigHolder():

    def __init__(self, config_path):

        self.config = json.load(tf.gfile.GFile(config_path, "r"))
