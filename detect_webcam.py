import os
import time
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils
from utils.category import coco_category_index

output_path = "data/outputs"

start_time = time.time()
model = tf.saved_model.load('data/models/efficientdet_d0_coco17_tpu-32_coco/saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

start_time = time.time()
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc("XVID")
out = cv2.VideoWriter(output_path, codec, fps, (width, height))

while True:
    _, img = cap.read()
    image_tensor = np.expand_dims(img, axis=0)
    detections = model(image_tensor)

    output_image = visualization_utils.visualize_boxes_and_labels_on_image_array(
        img.copy(),
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        coco_category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False)
    cv2.imshow("Object Detection", cv2.resize(output_image, (800, 600)))
    out.write(output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
end_time = time.time()
print('Elapsed time: ' + str(end_time - start_time) + 's')
