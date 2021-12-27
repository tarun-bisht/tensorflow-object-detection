from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
import os
import time
import albumentations as A
import tensorflow as tf
from utils import load_image, preprocess_image
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder


class ImageAugmentations:
    """
    image_size: resize image to -> (width, height)
    train_augs: include augmentations like random crop, rotation etc training if false then return
                only resize image as pytorch tensor
    """

    def __init__(
        self, image_size=None, apply_augs=False, normalize=False, bbox_format=None
    ):
        self.image_size = image_size
        self.apply_augs = apply_augs
        self.normalize = normalize
        self.bbox_format = bbox_format

    def train_augs(self):
        augs = []
        if self.image_size is not None:
            augs.append(A.Resize(self.image_size, self.image_size))
        if self.normalize:
            # imagenet normalization
            augs.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                )
            )
        if self.apply_augs:
            img_augs = [
                A.HorizontalFlip(p=0.5),
                A.ChannelShuffle(p=0.1),
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=0.2,
                            sat_shift_limit=0.2,
                            val_shift_limit=0.2,
                            p=0.5,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                A.ToGray(p=0.01),
                A.RandomGamma(p=0.1),
                A.Sharpen(p=0.1),
                A.Cutout(
                    num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.2
                ),
            ]
            augs = augs + img_augs
        if self.bbox_format is not None:
            return A.Compose(
                augs,
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    min_area=0,
                    min_visibility=0,
                    label_fields=["labels"],
                ),
            )
        return A.Compose(augs)

    def valid_augs(self):
        augs = []
        if self.image_size is not None:
            augs.append(A.Resize(self.image_size, self.image_size))
        if self.normalize:
            # imagenet normalization
            augs.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                )
            )
        if self.bbox_format is not None:
            return A.Compose(
                augs,
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    min_area=0,
                    min_visibility=0,
                    label_fields=["labels"],
                ),
            )
        return A.Compose(augs)


class DetectionDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        image_dir_path,
        labeltxt_path,
        batch_size,
        image_size=None,
        augmentations=None,
        img_preprocess_fn=None,
    ):
        super().__init__()
        self.df = df
        self.filenames = df["filename"].unique()
        self.image_dir_path = image_dir_path
        self.label_mapper, self.category_index = self.read_label_file(labeltxt_path)
        self.num_classes = (
            len(self.label_mapper) - 1
        )  # -1 for removing background class
        self.length = self.df.filename.nunique()
        self.batch_size = batch_size
        self.augs = augmentations
        self.image_size = image_size
        self.preprocess_fn = img_preprocess_fn

    def __len__(self):
        return self.length // self.batch_size

    def get_category_index(self):
        return self.category_index

    def get_label_mapper(self):
        return self.label_mapper

    def read_label_file(self, labeltxt_path):
        with open(labeltxt_path, "r") as label_file:
            lines = label_file.readlines()
            labels_to_index = {}
            index_to_label = {}
            for row, content in enumerate(lines):
                index_to_label[row] = {"id": row, "name": content.strip()}
                labels_to_index[content.strip()] = {"id": row, "name": content.strip()}
        return labels_to_index, index_to_label

    def __getitem__(self, idx):
        start_ = idx * self.batch_size
        end_ = (idx + 1) * self.batch_size
        return self.__get_samples(self.filenames[start_:end_])

    def __get_samples(self, filenames):
        images = []
        boxes = []
        onehots = []
        for filename in filenames:
            sample = self.df[self.df["filename"] == filename]
            file_path = os.path.join(self.image_dir_path, filename)
            image = load_image(file_path, size=self.image_size)
            # width and height
            width = sample.iloc[0, 1]
            height = sample.iloc[0, 2]
            # box coordinates
            xmin = sample.iloc[:, 4].values
            ymin = sample.iloc[:, 5].values
            xmax = sample.iloc[:, 6].values
            ymax = sample.iloc[:, 7].values
            # preparing boxes
            bboxes = np.transpose([xmin, ymin, xmax, ymax])
            # get text labels
            label_txt = sample.iloc[:, 3].values
            if self.augs is not None:
                for i in range(10):
                    transform = self.augs(image=image, bboxes=bboxes, labels=label_txt)
                    if len(transform["bboxes"]) > 0:
                        image = transform["image"]
                        bboxes = transform["bboxes"]
                        label_txt = transform["labels"]
                        break
            # one hot encoding labels
            label_id = [self.label_mapper[label]["id"] - 1 for label in label_txt]
            onehot = tf.one_hot(label_id, self.num_classes)
            # swapping boxes
            trans_bboxes = []
            for xmin, ymin, xmax, ymax in bboxes:
                trans_bboxes.append(
                    [ymin / height, xmin / width, ymax / height, xmax / width]
                )

            bboxes = tf.convert_to_tensor(trans_bboxes, dtype=tf.float32)
            # image tensor
            if self.preprocess_fn is not None:
                image = self.preprocess_fn(image)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            images.append(image)
            boxes.append(bboxes)
            onehots.append(onehot)
        return images, boxes, onehots


class Trainer:
    def __init__(
        self,
        batch_size,
        num_epochs,
        image_size,
        train_images_dir,
        val_images_dir,
        labelstxt_path,
        model_pipeline_path,
        model_name,
        model_checkpoint=None,
        early_stopping_patience=5,
        finetune=True,
    ):
        self.batch_size = batch_size
        self.model_name = model_name
        self.epochs = num_epochs
        self.patience = early_stopping_patience
        self.LOGGER = self.init_logger()
        self.image_size = image_size
        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir
        self.labelstxt_path = labelstxt_path
        self.finetune = finetune
        self.pipeline_config = model_pipeline_path
        self.checkpoint_path = model_checkpoint

    def init_logger(self, log_file="train.log"):
        os.makedirs("logs", exist_ok=True)
        log_file_path = os.path.join("log", log_file)
        logger = getLogger(__name__)
        logger.setLevel(INFO)
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        handler2 = FileHandler(filename=log_file_path)
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        return logger

    def create_detection_model(
        self,
        pipeline_config,
        num_classes,
        checkpoint_path=None,
        freeze_batchnorm=True,
        training=False,
        restore_classification_head=False,
    ):
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs["model"]
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = freeze_batchnorm
        detection_model = model_builder.build(
            model_config=model_config, is_training=training
        )

        if checkpoint_path is not None:
            fake_model_predictor = tf.compat.v2.train.Checkpoint(
                _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
                _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
            fake_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=fake_model_predictor,
            )
            ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
            ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 1024, 768, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)

        # save preprocessing steps here
        return detection_model

    def get_vars_to_finetune(self, detection_model):
        trainable_variables = detection_model.trainable_variables
        to_fine_tune = []
        # finetune box head and classification head ie.. train these layers only
        prefixes_to_train = [
            "WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead",
            "WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead",
        ]
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)
        return to_fine_tune

    @tf.function
    def train_step(self, train_ds, model, optimizer, vars_to_train, log_step=100):
        avg_loss = []
        step = 0
        for imgs, boxes, labels in train_ds:
            shapes = tf.constant(
                self.batch_size * [self.image_size[0], self.image_size[1], 3],
                dtype=tf.int32,
            )
            model.provide_groundtruth(
                groundtruth_boxes_list=boxes, groundtruth_classes_list=labels
            )
            with tf.GradientTape() as tape:
                imgs = tf.convert_to_tensor(imgs)
                # get predictions
                prediction_dict = model.predict(imgs, shapes)
                # calculate loss
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = (
                    losses_dict["Loss/localization_loss"]
                    + losses_dict["Loss/classification_loss"]
                )
                # calculate gradients
                gradients = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(gradients, vars_to_train))
            avg_loss.append(total_loss)
            step += 1
            if step % log_step == 0:
                self.LOGGER.info(f"train_loss: {np.mean(avg_loss):.4f}")
            return np.mean(avg_loss)

    @tf.function
    def val_step(self, val_ds, model):
        avg_loss = []
        for imgs, boxes, labels in val_ds:
            shapes = tf.constant(
                self.batch_size * [self.image_size[0], self.image_size[1], 3],
                dtype=tf.int32,
            )
            model.provide_groundtruth(
                groundtruth_boxes_list=boxes, groundtruth_classes_list=labels
            )
            imgs = tf.convert_to_tensor(imgs)
            prediction_dict = model.predict(imgs, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = (
                losses_dict["Loss/localization_loss"]
                + losses_dict["Loss/classification_loss"]
            )
            avg_loss.append(total_loss)
        return np.mean(avg_loss)

    def train_loop(
        self,
        train_df,
        valid_df,
        optimizer,
        log_step=100,
    ):
        tf.keras.backend.clear_session()
        # create dataset
        train_ds = DetectionDataset(
            train_df,
            image_dir_path=self.train_images_dir,
            labeltxt_path=self.labelstxt_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            img_preprocess_fn=preprocess_image,
        )
        val_ds = DetectionDataset(
            valid_df,
            image_dir_path=self.val_images_dir,
            labeltxt_path=self.labelstxt_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            img_preprocess_fn=preprocess_image,
        )
        num_classes = train_ds.num_classes
        # create model
        model = self.create_detection_model(
            self.pipeline_config, num_classes, self.checkpoint_path
        )
        if self.finetune:
            vars_to_train = self.get_vars_to_finetune(model)
        else:
            vars_to_train = model.trainable_variables
        # creating a checkpoint manager
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, f"{self.model_name}_best_ckpt", max_to_keep=1
        )
        best_loss = np.inf
        patience_count = 0
        for epoch in range(self.epochs):
            self.LOGGER.info(f"Epoch {epoch+1}")
            start_time = time.time()
            train_avg_loss = self.train_step(
                train_ds, model, optimizer, vars_to_train, log_step=log_step
            )
            val_avg_loss = self.val_step(val_ds, model)
            elapsed = time.time() - start_time
            self.LOGGER.info(
                f"Epoch {epoch+1} - train_loss: {train_avg_loss:.4f}  val_loss: {val_avg_loss:.4f}  time: {elapsed:.0f}s"
            )
            self.LOGGER.info(f"Epoch {epoch+1} - Loss: {val_avg_loss:.4f}")
            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                ckpt_manager.save()
                self.LOGGER.info(
                    f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model"
                )
                patience_count = 0
            else:
                patience_count += 1
            if patience_count > self.patience:
                break
