import colorsys
from random import random


class Object:
    counter = 0

    def __init__(self, roi, points, frame_no):
        self.id = Object.counter
        Object.counter += 1
        self.roi = roi
        self.roi_valid = True
        self.roi_history = {frame_no: roi}
        self.points = set(points)
        self.color = colorsys.hsv_to_rgb(random(), 1, 1)

    def add_match(self, roi, points, frame_no):
        self.roi = roi
        self.roi_valid = True
        self.roi_history[frame_no] = roi
        self.points = self.points.union(set(points))

    def predict_roi(self, frame_no):
        self.roi_valid = False
        pwh = [p.p-p.history[frame_no-1] for p in self.points if p.p is not None and (frame_no-1) in p.history]
        if len(pwh):
            move = sum(pwh)/len(pwh)
            self.roi[0] += move[1]
            self.roi[1] += move[0]
            self.roi[2] += move[1]
            self.roi[3] += move[0]

    def describe(self, frame_no):
        return '{},{},{},{},{},{},-1,-1,-1,-1\n'.format(frame_no + 1,
                                                        self.id,
                                                        self.roi[1],
                                                        self.roi[0],
                                                        self.roi[3] - self.roi[1],
                                                        self.roi[2] - self.roi[0])
