import os

import tensorflow as tf

from my_estimator import my_model_fn, predict_input_fn


class IrisClassifier(object):

    def __init__(self):
        model_dir = os.getenv('MODEL_DIR', './iris_model')

        # Create a custom estimator using my_model_fn to define the model
        tf.logging.info("Before classifier construction")
        estimator = tf.estimator.Estimator(
            model_fn=my_model_fn,
            model_dir=model_dir)  # Path to where checkpoints etc are stored
        self.estimator = estimator
        tf.logging.info("...done constructing classifier")

    def predict(self, input_data, feature_names):
        predict_results = self.estimator.predict(input_fn=lambda: predict_input_fn(input_data))

        return [[x["class_ids"]] for x in predict_results]

# if __name__ == '__main__':
#     t = IrisClassifier()
#     results = t.predict(np.array([[5.9, 3.0, 4.2, 1.5],
#                                   [6.9, 3.1, 5.4, 2.1],
#                                   [5.1, 3.3, 1.7, 0.5]]), None)
#     print(results)
