import argparse

import tensorflow as tf

from my_estimator import my_model_fn, my_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


def train_and_evaluate(main_args):
    EVAL_INTERVAL = 300  # seconds
    TRAIN_STEPS = 10000
    EVAL_STEPS = None

    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL,
                                        keep_checkpoint_max=3)

    # Create a custom estimator using my_model_fn to define the model
    tf.logging.info("Before classifier construction")
    estimator = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=main_args.output_path,
        config=run_config)  # Path to where checkpoints etc are stored
    tf.logging.info("...done constructing classifier")

    # 500 epochs = 500 * 120 records [60000] = (500 * 120) / 32 batches = 1875 batches
    # 4 epochs = 4 * 30 records = (4 * 30) / 32 batches = 3.75 batches

    # Train our model, use the previously function my_input_fn
    # Input to training is a file with training example
    # Stop training after 8 iterations of train data (epochs)

    tf.logging.info("Before classifier.train")
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: my_input_fn(main_args.input_train_path, 500, 256),
        max_steps=TRAIN_STEPS)

    tf.logging.info("...done classifier.train")

    # Evaluate our model using the examples contained in input_test_path
    # Return value will contain evaluation_metrics such as: loss & average_loss
    tf.logging.info("Before classifier.evaluate")
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: my_input_fn(main_args.input_test_path, 4),  # no need to batch in eval
        steps=EVAL_STEPS,
        start_delay_secs=60,  # start evaluating after N seconds
        throttle_secs=EVAL_INTERVAL,  # evaluate every N seconds
    )

    evaluate_result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    tf.logging.info("...done classifier.evaluate")
    tf.logging.info("Evaluation results")
    tf.logging.info(evaluate_result)


def main():
    # Parsing flags.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_path", type=str, default="")
    parser.add_argument("--input_test_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
