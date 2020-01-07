import os
from preprocessing import processor
import dataset.dataloader as data_loader
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

path = "/home/ktranrb/Desktop/image_classifier/dataset/train"

le = preprocessing.LabelEncoder()

p = processor.Processor(28, 28)

loader = data_loader.DataLoader(preprocessor=p)
data, label = loader.load(path, size=100)

le.fit(label)
labels = le.transform(label)

data = data.reshape(data.shape[0], 28*28*3)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=2)

model = KNeighborsClassifier(n_neighbors=2)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))
