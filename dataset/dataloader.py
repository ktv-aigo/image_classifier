import numpy as np
import os
import cv2

class DataLoader:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def load(self, path):
        entries = os.listdir(path)
        data, label = [], []
        for i in entries[:20]:
            img = cv2.imread(path+'/'+i)
            img = self.preprocessor.process(img)
            data.append(img)
            label.append(i[:3])
        return (np.asanyarray(data), np.asanyarray(label))
