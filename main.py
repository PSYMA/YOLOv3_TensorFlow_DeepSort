# coding: utf-8

from __future__ import division, print_function

import cv2  
import argparse
import numpy as np
import collections
import tensorflow as tf

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from sort.sort import *
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  

from model import yolov3

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    __slots__ = ()

parser = argparse.ArgumentParser()
parser.add_argument("source", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[320, 320],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--tracker", type=str, default='none', choices=["sort", "deep_sort"], 
                    help="tracking algorithm")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)
color_table = get_color_table(args.num_class)   

class ObjectDetection(object):
    def __init__(self) -> None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.555)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(self.sess, args.restore_path)
        self.cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source) 
        self.old_gray = None
        self.p0 = None
        self.speed = 0
        self.pixel_to_meter = 0.1

        if args.tracker == 'sort':
            self.tracker = Sort() 
        elif args.tracker == 'deep_sort':
            nn_budget = None
            max_cosine_distance = 0.4
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            self.tracker = Tracker(metric)
            deep_sort_model_file = 'deep_sort/resources/networks/mars-small128.pb'
            self.encoder = gdet.create_box_encoder(deep_sort_model_file, batch_size=1)

    def sort(self, boxes, frame, labels, scores):
        objs = []
        detections = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i] 
            detections.append([x, y, w, h])
            objs.append(Object(
                id = labels[i],
                score = scores[i] * 100,
                bbox = BBox(
                    xmin = x, ymin = y, xmax = w, ymax = h
                )
            ))
        
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        detections = np.array(detections)  
        if detections.any():
            if self.tracker != None:
                trdata = self.tracker.update(detections)
                if (np.array(trdata)).size:
                    for td in trdata: 
                        overlap = 0
                        x0, y0, x1, y1, track_id = int(td[0].item()), int(td[1].item()), int(td[2].item()), int(td[3].item()), int(td[4].item())   
                        for ob in objs:
                            dx0, dy0, dx1, dy1 = ob.bbox.xmin, ob.bbox.ymin, ob.bbox.xmax, ob.bbox.ymax
                            area = (min(dx1, x1) - max(dx0, x0)) * (min(dy1, y1) - max(dy0, y0))
                            if (area > overlap):
                                overlap = area
                                obj = ob
                        plot_one_box(frame, [x0, y0, x1, y1], label=str(track_id) + ", " + args.classes[obj.id] + ', {:.2f}%'.format(obj.score), color=color_table[obj.id])
                        
    def deep_sort(self, boxes, frame, labels, scores):
        names = []
        bboxes = []
        confidences = [] 
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            names.append(labels[i])
            bboxes.append([x, y, w - x, h - y])
            confidences.append(scores[i] * 100)   

        nms_max_overlap = 1.0
        features = self.encoder(frame, bboxes)  
        detections = [Detection(bbox, confidence, class_name, feature) for bbox, confidence, class_name, feature in zip(bboxes, confidences, names, features)]
        d_boxes = np.array([d.tlwh for d in detections])
        d_scores = np.array([d.confidence for d in detections])
        d_classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(d_boxes, nms_max_overlap, d_scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)   
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()
            name = track.class_name
            score = track.confidence

            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            plot_one_box(frame, [x0, y0, x1, y1], label=str(track_id) + ", " + args.classes[name] + ', {:.2f}%'.format(score), color=color_table[name])

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if args.letterbox_resize:
                image, resize_ratio, dw, dh = letterbox_resize(frame, args.new_size[0], args.new_size[1])
            else:
                height, width = frame.shape[:2]
                image = cv2.resize(frame, tuple(args.new_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image, np.float32)
            image = image[np.newaxis, :] / 255.
            boxes, scores, labels = self.sess.run([self.boxes, self.scores, self.labels], feed_dict={self.input_data: image})

            if args.letterbox_resize:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / resize_ratio
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes[:, [0, 2]] *= (width / float(args.new_size[0]))
                boxes[:, [1, 3]] *= (height / float(args.new_size[1]))
            
            if args.tracker == 'sort': 
                self.sort(boxes, frame, labels, scores) 
            elif args.tracker == 'deep_sort':
                self.deep_sort(boxes, frame, labels, scores)
            
            cv2.imshow('Object Detection', frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    od = ObjectDetection()
    od.run() 
