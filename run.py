#!/usr/bin/env python3

"""
Mask R-CNN + ORB tracker entry point.
"""

import argparse
import bz2
import glob
import math
import os
import pickle

import cv2
import numpy as np

from motion.tracker import Tracker
from plot import plot_image

DEFAULT_COCO_MODEL = 'mask_rcnn_coco.h5'


def __parse_args():
    """
    Parse arguments.

    :return: argument object
    """
    parser = argparse.ArgumentParser(description='Detect objects with Mask R-CNN and track them with ORB.')
    parser.add_argument('path', help='path to directory with frames')
    parser.add_argument('--file_mask', default='*.jpg')
    parser.add_argument('--output', '-o', default='result')
    parser.add_argument('--orb_points', type=int, default=8000)
    parser.add_argument('--orb_scale_factor', type=int, default=2)
    parser.add_argument('--fast_threshold', type=int, default=50)
    parser.add_argument('--orb_octaves', type=int)
    return parser.parse_args()


def run():
    """
    Run detection and tracking.
    """
    args = __parse_args()

    # Prepare output structure
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    full_path = os.path.join(args.output, 'full')
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    obj_path = os.path.join(args.output, 'objects')
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)

    # Get list of frames
    files = list(sorted(glob.glob(os.path.join(args.path, args.file_mask))))

    # Get first frame to set properties
    frame = cv2.imread(files[0])
    frame_height, frame_width = frame.shape[:2]

    # Create model object in inference mode
    # model = modellib.MaskRCNN("inference", CocoConfig())
    # model.load_weights(args.mrcnn_model, by_name=True)

    # model.detect([frame])
    # Compute number of octaves based on frame size and scale factor
    if not args.orb_octaves:
        args.orb_octaves = max(math.ceil(math.log(frame.shape[1] / 135, args.orb_scale_factor)), 1)

    # Create ORB detector
    orb = cv2.ORB.create(args.orb_points,
                         args.orb_scale_factor, args.orb_octaves,
                         31, 0, 2, cv2.ORB_HARRIS_SCORE,
                         31, args.fast_threshold)

    # Initialize tracker
    t = Tracker(args.orb_octaves, frame.shape[1], args.output)

    # Process files
    for f in files:
        # - load frame, convert from BGR to RGB
        frame = cv2.imread(f)[:, :, ::-1]
        # - convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # - detect and compute ORB PoIs with settings from `orb`
        pts, desc = orb.detectAndCompute(gray, None)
        # - load precomputed MaskR-CNN detections
        r = pickle.load(bz2.open(f.split('.')[0] + '.p.bz2', 'rb'))
        # - add frame (detections) to tracker
        do2oid = t.add_frame(pts, desc, r)
        # save image representations if required with --images argument
        plot_image(gray, t, args.output)

        bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)

        for detected_object, object_id in enumerate(do2oid):
            if object_id is None:
                continue

            bgra[:, :, 3] = r['masks'][:, :, detected_object] * 255

            roi = r['rois'][detected_object]
            h = roi[2] - roi[0]
            w = roi[3] - roi[1]
            size = int(max(w, h / 2))

            patch = np.zeros((size * 2, size, 4), np.uint8)

            c_h, c_w = int((roi[0] + roi[2]) / 2), int((roi[1] + roi[3]) / 2)
            pad_l = max(int(size / 2) - c_w, 0)
            pad_t = max(size - c_h, 0)
            pad_r = max(c_w + int(size / 2) - frame_width, 0)
            pad_b = max(c_h + size - frame_height, 0)
            valid_w = size - pad_l - pad_r
            valid_h = 2 * size - pad_t - pad_b
            img_l = max(int(c_w - size / 2), 0)
            img_t = max(int(c_h - size), 0)

            patch[pad_t:pad_t + valid_h, pad_l:pad_l + valid_w] = bgra[img_t:img_t + valid_h, img_l:img_l + valid_w]
            patch = cv2.resize(patch, (128, 256), interpolation=cv2.INTER_AREA)

            path = os.path.join(obj_path, str(object_id))
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(os.path.join(path, f'{t.frame_no}.png'), patch)


if __name__ == '__main__':
    run()
