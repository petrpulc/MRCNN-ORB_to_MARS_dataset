import numpy as np


class Point:
    def __init__(self, point, descriptor, frame_no):
        self.p = point
        self.d = np.array((0, 0))
        self.desc = descriptor
        self.history = {frame_no: point}

    def add_match(self, point, descriptor, upper_estimation, frame_no):
        self.d = point - self.p - upper_estimation
        self.p = point
        self.desc = descriptor
        self.history[frame_no] = point
