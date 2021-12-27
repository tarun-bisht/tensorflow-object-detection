import tensorflow as tf
import pandas as pd
from absl import flags
from src.train_utils import Trainer


flags.DEFINE_string("train_annotations_csv", None, "Path to train annotations csv file")
flags.DEFINE_string("test_annotations_csv", None, "Path to test annotations csv file")
flags.DEFINE_string("train_images_dir", None, "Path to train images directory")
flags.DEFINE_string("test_images_dir", None, "Path to test images directory")
flags.DEFINE_string("labelstxt_path", None, "Path to labelstxt file")
flags.DEFINE_string(
    "pipeline_config_path", None, "Path to model pipeline config " "file."
)
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to model checkpoints if not passed then model will train from scratch",
)
flags.DEFINE_integer("batch_size", 32, "Data batch size")
flags.DEFINE_integer("num_train_steps", 100, "Number of train steps.(epochs)")
flags.DEFINE_integer(
    "early_stopping_patience", 5, "number of steps to wait before early stopping"
)
flags.DEFINE_float("learning_rate", 0.01, "model training learning rate")
flags.DEFINE_string("model_name", None, "training model name used to store trained checkpoint")
flags.DEFINE_integer("image_width", 320, "image width for training model")
flags.DEFINE_integer("image_height", 320, "image height for training model")
flags.DEFINE_boolean(
    "finetune", True, "If True only finetune final layers else train whole model"
)
FLAGS = flags.FLAGS


def main(_argv):
    flags.mark_flag_as_required("train_annotations_csv")
    flags.mark_flag_as_required("test_annotations_csv")
    flags.mark_flag_as_required("train_images_dir")
    flags.mark_flag_as_required("test_images_dir")
    flags.mark_flag_as_required("labelstxt_path")
    flags.mark_flag_as_required("pipeline_config_path")
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("model_name")

    train_csv = pd.read_csv(FLAGS.train_annotations_csv)
    test_csv = pd.read_csv(FLAGS.test_annotations_csv)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=FLAGS.learning_rate, amsgrad=True
    )
    trainer = Trainer(
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_train_steps,
        image_size=(FLAGS.width, FLAGS.height),
        train_images_dir=FLAGS.train_images_dir,
        val_images_dir=FLAGS.test_images_dir,
        labelstxt_path=FLAGS.labeltxt_path,
        model_pipeline_path=FLAGS.pipeline_config_path,
        model_name=FLAGS.model_name,
        model_checkpoint=FLAGS.checkpoint_path,
        early_stopping_patience=FLAGS.early_stopping_patience,
        finetune=True,
    )
    trainer.train_loop(
        train_csv,
        test_csv,
        optimizer=optimizer,
        log_step=10,
    )


if __name__ == "__main__":
    tf.app.run()
