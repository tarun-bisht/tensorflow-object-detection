import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils
from utils.category import coco_category_index
from utils.utility import load_image

image_path = "data/images/cat.jpg"
output_path = "data/outputs"

img = load_image(image_path)

start_time = time.time()
tf.keras.backend.clear_session()
model = tf.saved_model.load('data/models/efficientdet_d0_coco17_tpu-32/saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

image_np = load_image(image_path)
image_tensor = np.expand_dims(image_np, axis=0)
start_time = time.time()
detections = model(image_tensor)
end_time = time.time()

output_image = visualization_utils.visualize_boxes_and_labels_on_image_array(
    image_np.copy(),
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.int32),
    detections['detection_scores'][0].numpy(),
    coco_category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.50,
    agnostic_mode=False)

print(output_image.shape)
output = Image.fromarray(output_image)
output.save(os.path.join(output_path, "output.jpg"))
output.show()
print('Elapsed time: ' + str(end_time - start_time) + 's')
