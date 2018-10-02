import tensorflow as tf

# The CSV features in our training & test data
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']


# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv, num_parallel_calls=4)  # Decode each line
               .cache()  # Warning: Caches entire dataset, can cause out of memory
               .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
               .repeat(repeat_count)  # Repeats dataset this # times
               .batch(32)
               .prefetch(1)  # Make sure you always have 1 batch ready to serve
               )
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def my_model_fn(
        features,  # This is batch_features from input_fn
        labels,  # This is batch_labels from input_fn
        mode):  # And instance of tf.estimator.ModeKeys, see below

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    # All our inputs are feature columns of type numeric_column
    feature_columns = [
        tf.feature_column.numeric_column(feature_names[0]),
        tf.feature_column.numeric_column(feature_names[1]),
        tf.feature_column.numeric_column(feature_names[2]),
        tf.feature_column.numeric_column(feature_names[3])
    ]

    # Create the layer of input
    input_layer = tf.feature_column.input_layer(features, feature_columns)

    # Definition of hidden layer: h1
    # We implement it as a fully-connected layer (tf.layers.dense)
    # Has 10 neurons, and uses ReLU as the activation function
    # Takes input_layer as input
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)

    # Definition of hidden layer: h2 (this is the logits layer)
    # Similar to h1, but takes h1 as input
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)

    # Output 'logits' layer is three number = probability distribution
    # between Iris Setosa, Versicolor, and Viginica
    logits = tf.layers.Dense(3)(h2)

    # class_ids will be the model prediction for the class (Iris flower type)
    # The output node with the highest value is our prediction
    predictions = {'class_ids': tf.argmax(input=logits, axis=1),
                   'probabilities': tf.nn.softmax(logits),
                   'logits': logits,
                   }

    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation and Training mode

    # Calculate the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Calculate the accuracy between the true labels, and our predictions
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'my_accuracy': accuracy})

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode

    # Default optimizer for DNNClassifier: Adagrad with learning rate=0.05
    # Our objective (train_op) is to minimize loss
    # Provide global step counter (used to count gradient updates)
    optimizer = tf.train.AdagradOptimizer(0.05)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step())

    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops
    tf.summary.scalar('my_accuracy', accuracy[1])

    # Return training operations: loss and train_op
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


def predict_input_fn(prediction_x):
    def decode(x):
        x = tf.split(x, 4)  # Need to split into our 4 features
        print(x)
        print("*********")
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_x)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels


# Let create a dataset for prediction
# We've taken the first 3 examples in FILE_TEST
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Setosa

# Predict all our prediction_input
# predict_results = classifier.predict(input_fn=lambda: new_input_fn(prediction_input))
#
# # Print results
# tf.logging.info("Predictions on memory")
# for idx, prediction in enumerate(predict_results):
