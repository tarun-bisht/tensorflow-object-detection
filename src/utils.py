from threading import Thread
import cv2
import os
import shutil
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from typing import Union


def load_image(path: str, size: tuple = None) -> np.array:
    """load an image from path specified and resize it to size specified

    Args:
        path (str): local path to image
        size (tuple, optional): width and height of image as tuple. Defaults to None.

    Returns:
        np.array: image as numpy array
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size:
        image = cv2.resize(image, size)
    return image.astype(np.uint8)


def load_url_image(url: str, size: tuple = None) -> np.array:
    """[summary]

    Args:
        url (str): url of image to load
        size (tuple, optional): width and height of image as tuple. Defaults to None.

    Returns:
        np.array: image as numpy array
    """
    img_request = urllib.request.urlopen(url)
    image = np.array(bytearray(img_request.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    if size:
        image = cv2.resize(image, size)
    return image


def load_labels(path: str) -> dict:
    """load label file into a dict

    Args:
        path (str): path to label file

    Returns:
        dict: dictionary representation of label files
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels = {}
        for row, content in enumerate(lines):
            labels[row] = {"id": row, "name": content.strip()}
    return labels


def preprocess_input(image: np.array) -> np.array:
    """preprocess image function

    Args:
        image (np.array): image as numpy array

    Returns:
        np.array: preprocesses image as numpy array
    """
    image = image.astype(np.float32)
    if image.shape[2] == 3:
        channel_means = [123.68, 116.779, 103.939]
        return (image - [[channel_means]]).astype(np.float32)
    else:
        return image.astype(np.float32)


def deprocess_image(preprocessed_image: np.array) -> np.array:
    """deprocess image function

    Args:
        preprocessed_image (np.array): preprocessed image as numpy array

    Returns:
        np.array: deprocessed image as numpy array
    """
    if preprocessed_image.shape[2] == 3:
        channel_means = [123.68, 116.779, 103.939]
        return (preprocessed_image + [[channel_means]]).astype(np.uint8)
    else:
        return preprocessed_image.astype(np.uint8)


def get_category_index(
    scores: np.array, classes: np.array, min_threshold: float = 0.5
) -> int:
    """return class index based on threshold and label file

    Args:
        scores (np.array): probability scores from detector
        classes (np.array): classes output from detector
        min_threshold (float, optional): threshold above which consider the labels. Defaults to 0.5.

    Returns:
        int: class index based on labels files
    """
    for i in range(len(scores)):
        if (scores[i] > min_threshold) and (scores[i] <= 1.0):
            return classes[i]
    return 0


def draw_boxes(
    image: np.array,
    boxes: np.array,
    classes: np.array,
    scores: np.array,
    category_index: dict,
    height: int,
    width: int,
    min_threshold: float = 0.5,
    put_label: bool = True,
) -> np.array:
    """Draw bounding box in images

    Args:
        image (np.array): image to draw rectangle on
        boxes (np.array): boxes to draw all elements are in range 0 and 1
        classes (np.array): class of each bounding box
        scores (np.array): probability score of each bounding box
        category_index (dict): dictionary representation of labels with id and class names
        height (int): height of image to scale bounding boxes with respect to image
        width (int): widht of image to scale bounding boxes with respect to image
        min_threshold (float, optional): min threshold above which to consider the bounding box. Defaults to 0.5.
        put_label (bool, optional): put class and score label at top of each bounding box. Defaults to True.

    Returns:
        np.array: image as numpy array with boxes drawn
    """
    for i in range(len(scores)):
        if (scores[i] > min_threshold) and (scores[i] <= 1.0):
            y_min = int(max(1, (boxes[i][0] * height)))
            x_min = int(max(1, (boxes[i][1] * width)))
            y_max = int(min(height, (boxes[i][2] * height)))
            x_max = int(min(width, (boxes[i][3] * width)))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)
            if put_label:
                if classes[i] in category_index:
                    class_name = category_index[classes[i]]["name"]
                else:
                    class_name = "N/A"
                label = f"{class_name}: {int(scores[i] * 100)}%"
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_min_y = max(y_min, labelSize[1] + 10)
                cv2.rectangle(
                    image,
                    (x_min, label_min_y - labelSize[1] - 10),
                    (x_min + labelSize[0], label_min_y + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    image,
                    label,
                    (x_min, label_min_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
    return image


class VideoStream:
    """Helper class to stream video in its own thread"""

    def __init__(
        self, resolution: tuple = (640, 480), cam: Union[int, str] = 0
    ) -> None:
        """class constructor

        Args:
            resolution (tuple, optional): resolution of video to stream. Defaults to (640, 480).
            cam (Union[int, str], optional): camera index or stream url. Defaults to 0.
        """
        self.stream = cv2.VideoCapture(cam)
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self) -> object:
        """start streaming

        Returns:
            object: VideoStream class object
        """
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self) -> None:
        """update loop it read from stream every frame"""
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self) -> np.array:
        """read next frame

        Returns:
            np.array: frame image as numpy array
        """
        return self.frame

    def stop(self) -> None:
        """stop stream and release stream object"""
        self.stopped = True


class SaveVideo:
    """Threaded utility to save video file from frames images"""

    def __init__(
        self,
        frames_path: str,
        output_path: str,
        fps: int,
        resolution: tuple = (640, 480),
    ) -> None:
        """Class Constructor

        Args:
            frames_path (str): path to frames folder
            output_path (str): path to save output video
            fps (int): fps of output video
            resolution (tuple, optional): resolution (widht and height) of output video. Defaults to (640, 480).
        """
        self.codec = cv2.VideoWriter_fourcc(*"XVID")
        self.frames_path = frames_path
        self.output_path = os.path.join(
            output_path, f"{os.path.basename(frames_path)}.avi"
        )
        self.resolution = resolution
        self.fps = fps

    def save(self) -> object:
        """start save thread

        Returns:
            object: SaveVideo class object
        """
        Thread(target=self.write, args=()).start()
        return self

    def write(self) -> None:
        """write video to output path specified inside a thread"""
        out = cv2.VideoWriter(self.output_path, self.codec, self.fps, self.resolution)
        for frame in sorted(os.listdir(self.frames_path)):
            frame_path = os.path.join(self.frames_path, frame)
            img_frame = cv2.imread(frame_path)
            out.write(img_frame)
        shutil.rmtree(self.frames_path)
        print(f"...Video file generated at {self.output_path}...")
        out.release()


def plot_detections(
    image_np, boxes, classes, scores, category_index, height, width, image_name=None
):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    draw_boxes(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        height,
        width,
        min_threshold=0.8,
    )
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)
