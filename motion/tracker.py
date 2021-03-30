import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from motion.object import Object
from .layers import PointLayer, HomographyLayer


class Tracker:
    """
    The object tracing class
    """

    def __init__(self, octaves, frame_width, sequence_name='result'):
        """
        Initialize the tracker.

        :param octaves: number of octaves
        :param frame_width: width of the frame to limit point candidate search to be relative to frame size
        :param sequence_name: path where to store the result to
        """
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

        self.file = open(os.path.join(sequence_name, 'det.txt'), 'w')

    def __del__(self):
        self.file.close()

    def add_frame(self, points, descriptors, detected_objects):
        """
        Add new frame to tracker

        :param points: list of cv2.Feature2D
        :param descriptors: list of point descriptors
        :param detected_objects: data from object detector
        """
        # First, sort our point by layer (octave) and relation to detected objects
        # - to do that we have to split points by to their layer (octave)
        octave_boundaries = [0]
        current_octave = 0

        # - find split points
        for i in range(len(points)):
            if points[i].octave != current_octave:
                octave_boundaries.append(i)
                current_octave += 1
                if current_octave + 1 == self.octaves:
                    break
        octave_boundaries.append(len(points))

        # - keep only point position and convert to int
        points = np.array([p.pt for p in points]).astype(int)

        # - sort points by detected object
        mask_points = {k: [] for k in range(len(detected_objects['class_ids']))}
        loose_points_idx = set()

        for i in range(len(points)):
            masks = np.argwhere(detected_objects['masks'][points[i, 1], points[i, 0]])
            if masks.size:
                for m in masks:
                    mask_points[m[0]].append(i)
            else:
                loose_points_idx.add(i)

        # Pass only loose points (intersection with loose points)
        # in up-most octave (between second-last and last boundary) to homography layer
        homography_points = list(
            set(range(octave_boundaries[-2], octave_boundaries[-1])).intersection(loose_points_idx))
        self.layers[-1].add_frame(np.take(points, homography_points, 0), np.take(descriptors, homography_points, 0))

        # Prepare an array that will hold relation of newly detected point to existing Point object
        point_objects = [None] * len(points)

        # Go from the highest layer (smallest resolution) down to full resolution
        # Add subset of points (ant their descriptors) to corresponding layer
        for i in reversed(range(self.octaves)):
            s, e = octave_boundaries[i:i + 2]
            point_objects[s:e] = self.layers[i].add_frame(points[s:e], descriptors[s:e], self.layers[i + 1])

        # Clear the list of loose points (not belonging to any object) from previous frame and fill with new list.
        self.loose_points = []
        for i in loose_points_idx:
            self.loose_points.append(point_objects[i])

        # Prepare a list of objects that are still live after introducing current frame
        live_objects = []

        # Then, match the detected objects
        # - mark non-person objects as used (to not be considered in matching)
        objects_to_match = [i for i in range(len(detected_objects['class_ids'])) if
                            detected_objects['class_ids'][i] == 1]
        otm_masks = np.take(detected_objects['masks'], objects_to_match, 2)
        otm_rois = np.take(detected_objects['rois'], objects_to_match, 0)

        # - compute matching cost from matched points
        cost = np.zeros((len(self.objects), len(objects_to_match)))
        # -- for each tracked object from last frame:
        for existing_object in range(len(self.objects)):
            # --- compute the number of Points that fall into mask of any object to match
            for pt in self.objects[existing_object].points:
                if pt.p is None:
                    continue
                for object_to_match in np.argwhere(otm_masks[pt.p[1], pt.p[0]]):
                    cost[existing_object, object_to_match] += 1
        # -- and normalize the number to a weight
        cost = 1 / np.log2(cost + 2) * 0.75

        # - add weight computed from RoI overlap
        # -- for each tracked object from last frame:
        for existing_object in range(len(self.objects)):
            # --- compute the overlap (intersection over union) with new detected object RoIs
            for object_to_match in range(len(objects_to_match)):
                obj = self.objects[existing_object].roi  # type: list
                otm = otm_rois[object_to_match]
                # TODO: transform RoIs with history of motion / layer estimation / homography
                overlap = max(0, min(obj[2], otm[2]) -
                              max(obj[0], otm[0])) * \
                          max(0, min(obj[3], otm[3]) -
                              max(obj[1], otm[1]))
                cost[existing_object, object_to_match] += 0.25 * (
                        1 - (overlap / (((obj[3] - obj[1]) * (obj[2] - obj[0]))
                                        + ((otm[3] - otm[1]) * (otm[2] - otm[0])) - overlap)))

        # - filter out tracked objects without candidate
        cost_row_ind = []  # row indexes to be considered in cost minimisation later
        for existing_object in range(len(self.objects)):
            obj_t = self.objects[existing_object]
            if sum(cost[existing_object]) < len(objects_to_match):
                # the cost is not maximal, therefore there is a possibility of match
                cost_row_ind.append(existing_object)
            else:
                # the existing object was not detected in new frame,
                # but we might still track it just with the tracked points
                if any(pt for pt in self.objects[existing_object].points):
                    obj_t.predict_roi(self.frame_no)
                    live_objects.append(obj_t)

        # - find the minimal mapping
        objects_with_no_match = set(range(len(objects_to_match)))
        if cost_row_ind:
            cost = np.take(cost, cost_row_ind, 0)
            for obj_idx, obj_candidate in zip(*linear_sum_assignment(cost)):
                # if cost is 1, no real match happened and we have to ignore it
                if cost[obj_idx, obj_candidate] == 1:
                    continue

                obj_t = self.objects[cost_row_ind[obj_idx]]

                obj_t.add_match(otm_rois[obj_candidate],
                                [point_objects[p] for p in mask_points[objects_to_match[obj_candidate]]],
                                self.frame_no)
                live_objects.append(obj_t)
                self.file.write(obj_t.describe(self.frame_no))
                objects_with_no_match.remove(obj_candidate)

        # Add objects that had no match as new ones to be tracked in future
        for new_object_id in objects_with_no_match:
            obj_t = Object(otm_rois[new_object_id],
                           [point_objects[p] for p in mask_points[objects_to_match[new_object_id]]],
                           self.frame_no)
            live_objects.append(obj_t)
            self.file.write(obj_t.describe(self.frame_no))

        # Store current set of live objects as the ones to be considered for tacking in next frame
        self.objects = live_objects

        self.frame_no += 1
