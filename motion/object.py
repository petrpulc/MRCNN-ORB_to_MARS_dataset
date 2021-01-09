import colorsys
from random import random


class Object:
    """
    Class holding information on tracked object
    """
    counter = 0  # global counter to give unique ID to the object

    def __init__(self, roi, points, frame_no):
        """
        Create new object to be tracked
        :param roi: Bounding box, position of top-left and bottom-right corner
                    (in this order, pixel 0,0 is in top left of the picture)
        :param points: Point objects associated with the object.
        :param frame_no: Frame number to index object position history.
        """
        # assign next object id
        self.id = Object.counter
        Object.counter += 1
        self.roi = roi
        self.roi_valid = True  # whether the RoI is acquired from object detection or predicted
        self.roi_history = {frame_no: roi}
        self.points = set(points)
        # Add random color to the object for visualization
        self.color = colorsys.hsv_to_rgb(random(), 1, 1)

    def add_match(self, roi, points, frame_no):
        """
        Add full match of the object in next frame from the object detection algorithm
        """
        self.roi = roi
        self.roi_valid = True
        self.roi_history[frame_no] = roi
        self.points.update(points)

    def predict_roi(self, frame_no):
        """
        Predict position of the object (bounding box) based on the tracked Points

        TODO: I believe that p.d should be used in combination with the upper estimation, but this has to be checked.
        """
        self.roi_valid = False
        pwh = [p.p - p.history[frame_no - 1] for p in self.points if p.p is not None and (frame_no - 1) in p.history]
        if len(pwh):
            move = sum(pwh) / len(pwh)
            self.roi[0] += move[1]
            self.roi[1] += move[0]
            self.roi[2] += move[1]
            self.roi[3] += move[0]

    def describe(self, frame_no):
        """
        Describe the object tracking in MOTChallenge format.
        """
        return '{},{},{},{},{},{},-1,-1,-1,-1\n'.format(frame_no + 1,
                                                        self.id,
                                                        self.roi[1],
                                                        self.roi[0],
                                                        self.roi[3] - self.roi[1],
                                                        self.roi[2] - self.roi[0])
