import logging
import logging.config
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from utils import cv_utils
from utils import operations as ops
from utils import tf_utils
import math

logging.config.fileConfig('logging.ini')

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'
OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5

def main():
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    with tf.Session(graph=detection_graph) as sess:
        processed_images = 0
        prev_cone1_xmax = None
        prev_cone2_xmin = None
        while True:
            # Wait for a new frame from the RealSense camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            if DETECT_EVERY_N_SECONDS:
                time.sleep(DETECT_EVERY_N_SECONDS)

            tic = time.time()

            # crops are images as ndarrays of shape
            # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
            # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
            #  the original image
            crops, crops_coordinates = ops.extract_crops(
                frame, CROP_HEIGHT, CROP_WIDTH,
                CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

            detection_dict = tf_utils.run_inference_for_batch(crops, sess)

            boxes = []
            for box_absolute, boxes_relative in zip(
                    crops_coordinates, detection_dict['detection_boxes']):
                boxes.extend(ops.get_absolute_boxes(
                    box_absolute,
                    boxes_relative[np.any(boxes_relative, axis=1)]))
            if boxes:
                boxes = np.vstack(boxes)

            boxes = ops.non_max_suppression_fast(
                boxes, NON_MAX_SUPPRESSION_THRESHOLD)

            boxes_scores = detection_dict['detection_scores']
            boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

            for box, score in zip(boxes, boxes_scores):
                if score > SCORE_THRESHOLD:
                    ymin, xmin, ymax, xmax = box
                    cone_color = cv_utils.predominant_rgb_color(frame, ymin, xmin, ymax, xmax)
                    cone_color_rgb = cone_color[::-1]
                    text = '{:.2f}'.format(score)
                    cv_utils.add_rectangle_with_text(
                        frame, ymin, xmin, ymax, xmax, cone_color_rgb, text)

            if len(boxes) >= 2:
                sorted_boxes = sorted(boxes, key=lambda box: box[1])
                cone1_xmax = sorted_boxes[0][3]
                cone2_xmin = sorted_boxes[1][1]

                if prev_cone1_xmax is not None and prev_cone2_xmin is not None:
                    prev_line_start = (int((prev_cone1_xmax + prev_cone2_xmin) / 2), frame.shape[0])
                    prev_line_end = (int((prev_cone1_xmax + prev_cone2_xmin) / 2), 0)

                    line_thickness = 2
                    line_color = (255, 0, 0)  # Blue color
                    cv2.line(frame, prev_line_start, prev_line_end, line_color, line_thickness)

                # Calculate the starting and ending points for the vertical line on the ground
                ground_y = 400  # Modify this value based on your ground position
                line_start = (int((cone1_xmax + cone2_xmin) / 2), ground_y)
                line_end = (int((cone1_xmax + cone2_xmin) / 2), 0)

                line_thickness = 2
                line_color = (0, 255, 0)  # Green color
                cv2.line(frame, line_start, line_end, line_color, line_thickness)

                prev_cone1_xmax = cone1_xmax
                prev_cone2_xmin = cone2_xmin

            if OUTPUT_WINDOW_WIDTH:
                frame = cv_utils.resize_width_keeping_aspect_ratio(
                    frame, OUTPUT_WINDOW_WIDTH)

            cv2.imshow('Detection result', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit if 'q' key is pressed
                break

            processed_images += 1

            toc = time.time()
            processing_time_ms = (toc - tic) * 100
            logging.debug(
                'Detected {} objects in {} images in {:.2f} ms'.format(
                    len(boxes), len(crops), processing_time_ms))

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
