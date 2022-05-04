
# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import sys

from PIL import Image

# Import utilites
from src.utils.card import label_map_util
from src.utils.card import visualization_utils as vis_util
import src.config as cfg

tf.disable_v2_behavior()

class INEReduce():
    def __init__(self):
        self.data = []
        
    def reduce(self, imageIn, imageOut, inDebug=False):
        NUM_CLASSES = 1
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(cfg.MODEL_LBLS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(cfg.MODEL_NAME, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                sess = tf.Session(graph=detection_graph)

                # Define input and output tensors (i.e. data) for the object detection classifier

                # Input tensor is the image
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Output tensors are the detection boxes, scores, and classes
                # Each box represents a part of the image where a particular object was detected
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represents level of confidence for each of the objects.
                # The score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

                # Number of objects detected
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Load image using OpenCV and
                # expand image dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                image = cv2.imread(imageIn)
                
                shape = np.shape(image)
                im_width, im_height = shape[1], shape[0]
                                
                if im_width > im_height:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    #cv2.imshow("Rotated by -90 Degrees", rotated)
                    #cv2.waitKey()
                    image = rotated                
                                
                image_expanded = np.expand_dims(image, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})
                # Draw the results of the detection (aka 'visulaize the results')
                image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=0.60)

                ymin, xmin, ymax, xmax = array_coord

                shape = np.shape(image)
                im_width, im_height = shape[1], shape[0]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                
                print("Original -> Alto {}, Ancho {}".format(im_width, im_height))
                print("Izquierda {}, Derecho {}, Superior {}, Abajo {}".format(left, right, top, bottom))

                # Using Image to crop and save the extracted copied image
                im = Image.open(imageIn)
                im.crop((left, top, right, bottom)).save(imageOut, quality=95)

                if inDebug == True:
                    cv2.imshow('ID-CARD-DETECTOR : ', image)
                    image_cropped = cv2.imread(imageOut)
                    cv2.imshow("ID-CARD-CROPPED : ", image_cropped)
    
                    # All the results have been drawn on image. Now display the image.
                    #cv2.imshow('ID CARD DETECTOR', image)
                    # Press any key to close the image
                    cv2.waitKey(0)
    
                    # Clean up
                    cv2.destroyAllWindows()
