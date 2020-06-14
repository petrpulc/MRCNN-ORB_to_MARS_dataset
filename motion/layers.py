from typing import Union, List

import cv2
import numpy as np
from scipy.spatial import cKDTree

# from plot import plot_matches
from motion.point import Point


class PointLayer:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_vectors = None
        self.last_vectors_tree = None  # type: Union[None, cKDTree]
        self.points = []

    def add_frame(self, points, descriptors, layer_above):
        points_index = cKDTree(points)
        used = [None] * len(points)  # type: List[Union[None, Point]]

        self.last_vectors = []
        live_points = []

        for f in self.points:  # type: Point
            match_found = False

            upper_estimation = layer_above.estimate_delta(f.p)
            estimated = f.p + f.d + upper_estimation
            candidates = points_index.query_ball_point(estimated, 0.01 * self.tracker.frame_width)
            candidates = {i: cv2.norm(f.desc, descriptors[i], cv2.NORM_HAMMING) for i in candidates}
            for c, d in sorted(candidates.items(), key=lambda i: i[1]):
                if d > 64:
                    # Points too bad.
                    break
                if used[c]:
                    # Point already used.
                    continue

                match_found = True
                used[c] = f
                self.last_vectors.append((f.p, points[c] - f.p))
                f.add_match(points[c], descriptors[c], upper_estimation, self.tracker.frame_no)
                live_points.append(f)
                break

            if not match_found:
                f.p = None

        self.points = live_points

        if self.last_vectors:
            self.last_vectors_tree = cKDTree([x[0] for x in self.last_vectors])

        for i in range(len(points)):
            if not used[i]:
                point = Point(points[i], descriptors[i], self.tracker.frame_no)
                used[i] = point
                self.points.append(point)

        return used

    def estimate_delta(self, point):
        if self.last_vectors_tree is None:
            return np.array((0, 0))

        v = [self.last_vectors[i][1] for i in self.last_vectors_tree.query(point, 3)[1] if i < len(self.last_vectors)]
        if len(v):
            return sum(v) / len(v)
        else:
            return np.array((0, 0))


class HomographyLayer:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_frame_points = None
        self.last_frame_desc = None
        self.matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, True)  # type: cv2.BFMatcher
        self.homography = np.eye(3)

    def add_frame(self, points, descriptors):
        if self.last_frame_points is not None:
            # Match points based on descriptions
            matches = self.matcher.match(descriptors, self.last_frame_desc)

            # matches = sorted(matches, key=lambda x: x.distance)

            # Filter good matches (threshold of 64) and store corresponding points
            last_frame_matched = np.take(self.last_frame_points,
                                         [m.trainIdx for m in matches if m.distance <= 40], axis=0)
            new_frame_matched = np.take(points,
                                        [m.queryIdx for m in matches if m.distance <= 40], axis=0)

            # plot_matches(cv2.imread('000002.jpg'), last_frame_matched, new_frame_matched)

            # Find homography, identity if fails
            if last_frame_matched.size == 0:
                self.homography = np.eye(3)
            else:
                self.homography, _ = cv2.findHomography(last_frame_matched, new_frame_matched, cv2.RANSAC)

        # Store point positions and descriptors
        self.last_frame_points = points
        self.last_frame_desc = descriptors

    def estimate_delta(self, point):
        return np.dot(self.homography, np.array((point[0], point[1], 1)))[:2] - point
