import os

import numpy as np

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

        # Then, match the detected objects
        # Mark non-person objects as used
        used_objects = [class_id != 1 for class_id in detected_objects['class_ids']]

        live_objects = []
        for obj in self.objects:  # type: Object
            candidate = None
            live = False

            if obj.points:
                candidate_objects = []
                for pt in obj.points:  # type: Point
                    if pt.p is not None:
                        live = True
                        for i in np.argwhere(detected_objects['masks'][pt.p[1], pt.p[0]]):
                            if not used_objects[i[0]]:
                                candidate_objects.append(i[0])
                if len(candidate_objects) > 3 * len(set(candidate_objects)):
                    candidate = max(set(candidate_objects), key=candidate_objects.count)

            if candidate is None and obj.roi_valid:
                # rois in shape y,x,y,x
                bottom_left = obj.roi[:2] + self.layers[-1].estimate_delta(obj.roi[:2][::-1])[::-1]
                top_right = obj.roi[2:] + self.layers[-1].estimate_delta(obj.roi[2:][::-1])[::-1]
                candidate_overlap = 0
                for i in range(len(detected_objects['class_ids'])):
                    overlap = max(0, min(detected_objects['rois'][i, 2], top_right[0]) -
                                  max(detected_objects['rois'][i, 0], bottom_left[0])) * \
                              max(0, min(detected_objects['rois'][i, 3], top_right[1]) -
                                  max(detected_objects['rois'][i, 1], bottom_left[1]))
                    if overlap > candidate_overlap and not used_objects[i]:
                        candidate = i
                        candidate_overlap = overlap

            if candidate is not None:
                live_objects.append(obj)
                obj.add_match(detected_objects['rois'][candidate],
                              [point_objects[p] for p in mask_points[candidate]],
                              self.frame_no)
                used_objects[candidate] = True
                self.file.write(obj.describe(self.frame_no))
            elif live:
                obj.predict_roi(self.frame_no)
                # self.file.write(obj.describe(self.frame_no))
                live_objects.append(obj)

        self.objects = live_objects

        for i in range(len(detected_objects['class_ids'])):
            if not used_objects[i]:
                obj = Object(detected_objects['rois'][i],
                             [point_objects[p] for p in mask_points[i]],
                             self.frame_no)
                self.objects.append(obj)
                self.file.write(obj.describe(self.frame_no))

        self.frame_no += 1
