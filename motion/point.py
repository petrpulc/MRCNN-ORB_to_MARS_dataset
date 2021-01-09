import numpy as np


class Point:
    """
    Point tracking object.
    """

    def __init__(self, point, descriptor, frame_no):
        """
        Create Point object.

        :param point: position of point of interest
        :param descriptor: point of interest descriptor
        :param frame_no: frame number to keep in history
        """
        self.d = np.array((0, 0))  # no motion estimated on point creation
        self.p = point
        self.desc = descriptor
        self.history = {frame_no: point}

    def add_match(self, point, descriptor, upper_estimation, frame_no):
        """
        Add point match to trajectory.

        :param point: new point position
        :param descriptor: new point descriptor (the description is allowed to shift)
        :param upper_estimation: movement estimation from upper layer (to cancel out relative motion)
        :param frame_no: frame number to keep in history
        """
        self.d = point - self.p - upper_estimation
        self.p = point
        self.desc = descriptor
        self.history[frame_no] = point
