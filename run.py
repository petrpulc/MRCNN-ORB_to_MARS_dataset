#!/usr/bin/env python3

"""
Mask R-CNN + ORB tracker entry point.
"""

import argparse
import glob
import os

import cv2

# import mrcnn.model as modellib
from motion.tracker import Tracker
# from mrcnn.config import CocoConfig
# from mrcnn.utils import download_trained_weights
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
    parser.add_argument('--fast_threshold', type=float, default=50)
    parser.add_argument('--orb_octaves', type=int, default=3)
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
        os.mkdir(args.output)
    if args.images:
        for sub in ('mrcnn', 'rois', 'full'):
            full_path = os.path.join(args.output, sub)
            if not os.path.exists(full_path):
                os.mkdir(full_path)

    # Get list of frames
    files = list(sorted(glob.glob(os.path.join(args.path, args.file_mask))))

    # Get first frame to set properties
    frame = cv2.imread(files[0])

    # Create model object in inference mode
#    model = modellib.MaskRCNN("inference", CocoConfig())
#    model.load_weights(args.mrcnn_model, by_name=True)

#    model.detect([frame])

    # Create ORB detector
    orb = cv2.ORB.create(args.orb_points, 2, args.orb_octaves, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, args.fast_threshold)

    # Initialize tracker
    t = Tracker(args.orb_octaves, frame.shape[1], args.output)

    # Process files
    for f in files:
        frame = cv2.imread(f)[:, :, ::-1]
        cpu_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pts, desc = orb.detectAndCompute(cpu_gray, None)
#        r = model.detect([frame])[0]
        r = pickle.load(bz2.open(f.split('.')[0]+'.p.bz2', 'rb'))
        t.add_frame(pts, desc, r)
        if args.images:
            plot_mrcnn(cpu_gray, r, t, args.output)
            plot_image(cpu_gray, t, args.output)


if __name__ == '__main__':
    run()
