import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from motion.object import Object
from motion.point import Point
from .layers import PointLayer, HomographyLayer


class Tracker:
    def __init__(self, octaves, frame_width, seqname='result'):
        self.octaves = octaves
        self.layers = []
        self.objects = []

        # Initialize same number of Point layers as octaves
        for o in range(self.octaves):
            self.layers.append(PointLayer(self))

        # Add Homography layer on top
        self.layers.append(HomographyLayer(self))

        self.frame_no = 0
        self.frame_width = frame_width

        self.loose_points = []

        self.file = open(os.path.join(seqname, 'det.txt'), 'w')

    def __del__(self):
        self.file.close()

    def add_frame(self, points, descriptors, detected_objects):
        # First, match the points of interest across all layers
        octave_boundaries = [0]
        current_octave = 0

        # Find split points
        for i in range(len(points)):
            if points[i].octave != current_octave:
                octave_boundaries.append(i)
                current_octave += 1
                if current_octave + 1 == self.octaves:
                    break
        octave_boundaries.append(len(points))

        # Keep only point position and convert to int
        points = np.array([p.pt for p in points]).astype(int)

        # Sort points by detected object
        mask_points = {k: [] for k in range(len(detected_objects['class_ids']))}
        loose_points_idx = set()

        for i in range(len(points)):
            masks = np.argwhere(detected_objects['masks'][points[i, 1], points[i, 0]])
            if masks.size:
                for m in masks:
                    mask_points[m[0]].append(i)
            else:
                loose_points_idx.add(i)

        # Pass only loose points in up-most octave to homography layer
        homography_points = list(
            set(range(octave_boundaries[-2], octave_boundaries[-1])).intersection(loose_points_idx))
        self.layers[-1].add_frame(np.take(points, homography_points, 0), np.take(descriptors, homography_points, 0))

        point_objects = [None] * len(points)

        for i in reversed(range(self.octaves)):
            s, e = octave_boundaries[i:i + 2]
            point_objects[s:e] = self.layers[i].add_frame(points[s:e], descriptors[s:e], self.layers[i + 1])

        self.loose_points = []
        for i in loose_points_idx:
            self.loose_points.append(point_objects[i])

        live_objects = []

        # Then, match the detected objects
        # Mark non-person objects as used
        objects_to_match = [i for i in range(len(detected_objects['class_ids'])) if
                            detected_objects['class_ids'][i] == 1]
        otm_masks = np.take(detected_objects['masks'], objects_to_match, 2)
        otm_rois = np.take(detected_objects['rois'], objects_to_match, 0)

        # cost from paths
        cost = np.zeros((len(self.objects), len(objects_to_match)))
        for x in range(len(self.objects)):
            for pt in self.objects[x].points:
                if pt.p is None:
                    continue
                for y in np.argwhere(otm_masks[pt.p[1], pt.p[0]]):
                    cost[x, y] += 1
        cost = 1 / np.log2(cost + 2) * 0.75

        for x in range(len(self.objects)):
            for y in range(len(objects_to_match)):
                obj = self.objects[x].roi  # type: list
                otm = otm_rois[y]
                overlap = max(0, min(obj[2], otm[2]) -
                              max(obj[0], otm[0])) * \
                          max(0, min(obj[3], otm[3]) -
                              max(obj[1], otm[1]))
                cost[x, y] += 0.25 * (1 - (overlap / (((obj[3] - obj[1]) * (obj[2] - obj[0]))
                                                      + ((otm[3] - otm[1]) * (otm[2] - otm[0])) - overlap)))

        # filter out tracked objects without candidate
        cost_rows = []

        for x in range(len(self.objects)):
            obj_t = self.objects[x]
            if sum(cost[x]) < len(objects_to_match):
                cost_rows.append(x)
            else:
                if any(pt for pt in self.objects[x].points):
                    obj_t.predict_roi(self.frame_no)
                    live_objects.append(obj_t)

        col_ind = []

        if cost_rows:
            cost = np.take(cost, cost_rows,0)
            row_ind, col_ind = linear_sum_assignment(cost)
            for obj_idx, obj_candidate in zip(row_ind, col_ind):
                obj_t = self.objects[cost_rows[obj_idx]]
                obj_t.add_match(otm_rois[obj_candidate],
                                [point_objects[p] for p in mask_points[objects_to_match[obj_candidate]]],
                                self.frame_no)
                live_objects.append(obj_t)
                self.file.write(obj_t.describe(self.frame_no))

        for new_object_id in set(range(len(objects_to_match)))-set(col_ind):
            obj_t = Object(otm_rois[new_object_id],
                         [point_objects[p] for p in mask_points[objects_to_match[new_object_id]]],
                         self.frame_no)
            live_objects.append(obj_t)
            self.file.write(obj_t.describe(self.frame_no))

        self.objects = live_objects

        self.frame_no += 1
