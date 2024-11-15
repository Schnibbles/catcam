import argparse
import sys
import time
from typing import List
import socket
import argparse
import sys
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

import cv2
import numpy as np
start_time = time.time()
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

last_detections = []
LABELS = None



PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="1920" height="1080" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    import argparse
    import sys
    from functools import lru_cache

    import cv2
    import numpy as np

    from picamera2 import MappedArray, Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (NetworkIntrinsics,
                                          postprocess_nanodet_detection)

    last_detections = []

class Detection:
        def __init__(self, coords, category, conf, metadata):
            """Create a Detection object, recording the bounding box, category and confidence."""
            self.category = category
            self.conf = conf
            self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
        """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
        global last_detections
        bbox_normalization = intrinsics.bbox_normalization
        bbox_order = intrinsics.bbox_order
        threshold = args.threshold
        iou = args.iou
        max_detections = args.max_detections

        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = imx500.get_input_size()
        if np_outputs is None:
            return last_detections
        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                              max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        last_detections = [
            Detection(box, category, score, metadata)
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]
        return last_detections

@lru_cache
def get_labels():
        labels = intrinsics.labels

        if intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

def draw_detections(request, stream="main"):
        """Draw the detections for this request onto the ISP output."""
        detections = last_results
        if detections is None:
            return
        labels = get_labels()
        with MappedArray(request, stream) as m:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y + 15

                # Create a copy of the array to draw the background with opacity
                overlay = m.array.copy()

                # Draw the background rectangle on the overlay
                cv2.rectangle(overlay,
                              (text_x, text_y - text_height),
                              (text_x + text_width, text_y + baseline),
                              (255, 255, 255),  # Background color (white)
                              cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

                # Draw text on top of the background
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

            if intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))



def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, help="Path of the model",
                            default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
        parser.add_argument("--fps", type=int, help="Frames per second")
        parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
        parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                            help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
        parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
        parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
        parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
        parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
        parser.add_argument("--postprocess", choices=["", "nanodet"],
                            default=None, help="Run post process of type")
        parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                            help="preserve the pixel aspect ratio of the input tensor")
        parser.add_argument("--labels", type=str,
                            help="Path to the labels file")
        parser.add_argument("--print-intrinsics", action="store_true",
                            help="Print JSON network_intrinsics then exit")
        return parser.parse_args()

if __name__ == "__main__":
        args = get_args()

        # This must be called before instantiation of Picamera2
        imx500 = IMX500(args.model)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            print("Network is not an object detection task", file=sys.stderr)
            exit()

        # Override intrinsics from args
        for key, value in vars(args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    intrinsics.labels = f.read().splitlines()
            elif hasattr(intrinsics, key) and value is not None:
                setattr(intrinsics, key, value)

        # Defaults
        if intrinsics.labels is None:
            with open("assets/coco_labels.txt", "r") as f:
                intrinsics.labels = f.read().splitlines()
        intrinsics.update_with_defaults()

        if args.print_intrinsics:
            print(intrinsics)
            exit()

        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_video_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

        imx500.show_network_fw_progress_bar()
        picam2.configure(config)

        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()


        picam2.pre_callback = draw_detections
        output = StreamingOutput()

        picam2.start_recording(JpegEncoder(), FileOutput(output))

        while True:
            try:
                last_results = None
                address = ('', 8000)
                server = StreamingServer(address, StreamingHandler)
                server.handle_request()
                last_results = parse_detections(picam2.capture_metadata())
            finally:
                picam2.stop_recording()

