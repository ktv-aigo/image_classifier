import cv2

class Processor:
    def __init__(self,height, width):
        self.height = height
        self.width = width

    def process(self, image):
        return cv2.resize(image,(self.width, self.height))

