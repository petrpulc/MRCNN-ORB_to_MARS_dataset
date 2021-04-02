from typing import Union, List

import cv2
import numpy as np
from scipy.spatial import cKDTree

# from plot import plot_matches
from motion.point import Point


class PointLayer:
    """
    Point of interest registration (and prediction) layer that uses the
    registered motion as a direct source of prediction.
    """

    def __init__(self, tracker):
        """
        Initialize empty point cloud.
        """
        self.tracker = tracker
        self.last_vectors = None
        self.last_vectors_tree = None  # type: Union[None, cKDTree]
        self.points = []

    def add_frame(self, points, descriptors, layer_above):
        """
        Register new set of point of interest from an incoming frame using a motion prediction
        from movement history combined with prediction from a layer above the current one.

        :param points: set of points to register
        :param descriptors: descriptors used for registration
        :param layer_above: layer used for movement prediction in current frame
        :returns: list of Point objects corresponding to detected PoIs
        """
        # Create position index and information on usage of points in the incoming frame
        points_index = cKDTree(points)
        used = [None] * len(points)  # type: List[Union[None, Point]]

        # Clear array of vectors used for prediction by layer below
        self.last_vectors = []
        # Clear array of currently live points from previous frame
        live_points = []

        # For each point registered in previous frame
        for f in self.points:  # type: Point
            match_found = False

            # - get motion estimation from layer above
            upper_estimation = layer_above.estimate_delta(f.p)
            # - propose point position in incoming frame as combination of last known position, last known motion
            # and motion estimation in layer above
            estimated = f.p + f.d + upper_estimation
            # - get list of candidate points in incoming frame (its index) in a 0.01*frame_width radius around
            # the new predicted position
            candidates = points_index.query_ball_point(estimated, 0.01 * self.tracker.frame_width)
            # - compute Hamming distance on point descriptors as the qualitative distance
            candidates = {i: cv2.norm(f.desc, descriptors[i], cv2.NORM_HAMMING) for i in candidates}
            # - find unused candidate point with smallest Hamming distance
            # TODO: consider point angle on matching
            for c, d in sorted(candidates.items(), key=lambda i: i[1]):
                if d > 64:
                    # Points too bad.
                    break
                if used[c]:
                    # Point already used.
                    continue

                match_found = True
                # - once found:
                # -- mark as used, store registered Point object
                used[c] = f
                # -- append motion vector to list used for prediction in layer below
                self.last_vectors.append((f.p, points[c] - f.p))
                # -- add match to the registered point
                f.add_match(points[c], descriptors[c], upper_estimation, self.tracker.frame_no)
                # -- add point to set of live points - that can be tracked further
                live_points.append(f)
                break

            if not match_found:
                f.p = None

        self.points = live_points

        # TODO: visualize matches?

        # construct index of registered motion vectors for estimation in layer below
        if self.last_vectors:
            self.last_vectors_tree = cKDTree([x[0] for x in self.last_vectors])

        # create new Point objects for all detected PoIs that were not registered
        for i in range(len(points)):
            if not used[i]:
                point = Point(points[i], descriptors[i], self.tracker.frame_no)
                used[i] = point
                self.points.append(point)

        return used

    def estimate_delta(self, point):
        """
        Motion estimation for given point based on registered points of interest.

        :param point: coordinates to make motion prediction for
        :returns: predicted motion
        """
        if self.last_vectors_tree is None:
            # return zero prediction if no index available (first frame)
            return np.array((0, 0))

        # TODO: point weighted by distance?
        v = [self.last_vectors[i][1] for i in self.last_vectors_tree.query(point, 3)[1] if i < len(self.last_vectors)]
        # get up to 3 points closest to query point
        if len(v):
            # return average prediction
            return sum(v) / len(v)
        else:
            # return zero if no point found
            return np.array((0, 0))


class HomographyLayer:
    """
    Motion prediction layer that registers points of interest to a homography.
    Used as a top-most layer for crude model of camera motion.
    """

    def __init__(self, tracker):
        """
        Initialize empty homography registration.
        """
        self.tracker = tracker
        self.last_frame_points = None
        self.last_frame_desc = None
        self.matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, True)  # type: cv2.BFMatcher
        self.homography = np.eye(3)

    def add_frame(self, points, descriptors):
        """
        Register new set of point of interest from an incoming frame and deduce homography
        that is later used as a high-level motion estimation.

        :param points: set of points to register
        :param descriptors: descriptors used for registration
        """
        if self.last_frame_points is None or not self.last_frame_desc.size or not descriptors.size:
            self.homography = np.eye(3)
        else:
            # Match points based on descriptions
            # TODO: check bruteforce matcher approach
            matches = self.matcher.match(descriptors, self.last_frame_desc)

            # matches = sorted(matches, key=lambda x: x.distance)

            # Filter good matches (threshold of 40) and store corresponding points
            last_frame_matched = np.take(self.last_frame_points,
                                         [m.trainIdx for m in matches if m.distance <= 40], axis=0)
            new_frame_matched = np.take(points,
                                        [m.queryIdx for m in matches if m.distance <= 40], axis=0)

            # TODO: make debug output an argument of executable, improve display of matches
            #  (anaglyph with matches and homography vizualisation)
            # plot_matches(cv2.imread('000002.jpg'), last_frame_matched, new_frame_matched)

            # Find homography, identity if fails
            if last_frame_matched.size == 0:
                self.homography = np.eye(3)
            else:
                # TODO: verify the ransac matching efficacy
                self.homography, _ = cv2.findHomography(last_frame_matched, new_frame_matched, cv2.RANSAC)
            if self.homography is None:
                self.homography = np.eye(3)

        # Store point positions and descriptors
        self.last_frame_points = points
        self.last_frame_desc = descriptors

    def estimate_delta(self, point):
        """
        Motion estimation for given point based on deduced homography.

        :param point: coordinates to make motion prediction for
        :returns: predicted motion
        """
        return np.dot(self.homography, np.array((point[0], point[1], 1)))[:2] - point
