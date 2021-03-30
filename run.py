#!/usr/bin/env python3

"""
Mask R-CNN + ORB tracker entry point.
"""

import argparse
import glob
import math
import os

import cv2

import mrcnn.model as modellib
from motion.tracker import Tracker
from mrcnn.config import CocoConfig
from mrcnn.utils import download_trained_weights
from plot import plot_mrcnn, plot_image
import pickle, bz2

DEFAULT_COCO_MODEL = 'mask_rcnn_coco.h5'


def __parse_args():
    """
    Parse arguments.

    :return: argument object
    """
    parser = argparse.ArgumentParser(description='Detect objects with Mask R-CNN and track them with ORB.')
    parser.add_argument('path', help='path to directory with frames')
    parser.add_argument('--file_mask', default='*.jpg')
    parser.add_argument('--mrcnn_model', default=DEFAULT_COCO_MODEL,
                        help='path to mrcnn model, default is downloaded automatically if not available')
    parser.add_argument('--output', '-o', default='result')
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--orb_points', type=int, default=5000)
    parser.add_argument('--orb_scale_factor', type=int, default=2)
    parser.add_argument('--fast_threshold', type=int, default=50)
    parser.add_argument('--orb_octaves', type=int)
    return parser.parse_args()


def __check_model(path):
    """
    Check if model file available, download if default is missing.

    Based on: https://github.com/matterport/Mask_RCNN

    :param path: path to model
    """
    if os.path.exists(path):
        return

    if os.path.basename(path) != DEFAULT_COCO_MODEL:
        raise FileNotFoundError('Unknown model, please download manually...')

    download_trained_weights(path)


def run():
    """
    Run detection and tracking.
    """
    args = __parse_args()
    __check_model(args.mrcnn_model)

    # Prepare output structure
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    if args.images:
        for sub in ('mrcnn', 'rois', 'full'):
            full_path = os.path.join(args.output, sub)
            if not os.path.exists(full_path):
                os.mkdir(full_path)
    obj_path = os.path.join(args.output, 'objects')
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)

    # Get list of frames
    files = list(sorted(glob.glob(os.path.join(args.path, args.file_mask))))

    # Get first frame to set properties
    frame = cv2.imread(files[0])

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
        # - run detection with MaskR-CNN on full color frame
        # r = model.detect([frame])[0]
        # - alternatively, load precomputed MaskR-CNN detections
        r = pickle.load(bz2.open(f.split('.')[0] + '.p.bz2', 'rb'))
        # - add frame (detections) to tracker
        do2oid = t.add_frame(pts, desc, r)
        # save image representations if required with --images argument
        if args.images:
            # plot_mrcnn(gray, r, t, args.output)
            plot_image(gray, t, args.output)
        for detected_object, object_id in enumerate(do2oid):
            if object_id is None:
                continue
            bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            bgra[:, :, 3] = r['masks'][:, :, detected_object] * 255

            roi = r['rois'][detected_object]
            size = int(max(roi[2] - roi[0], roi[3] - roi[1]) / 2)
            centroid = (int((roi[0] + roi[2]) / 2), int((roi[1] + roi[3]) / 2))
            patch = bgra[centroid[0] - size:centroid[0] + size, centroid[1] - size:centroid[1] + size]
            patch = cv2.resize(patch, (50, 50), interpolation=cv2.INTER_AREA)

            path = os.path.join(obj_path, str(object_id))
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(os.path.join(path, f'{t.frame_no}.png'), patch)


if __name__ == '__main__':
    run()
