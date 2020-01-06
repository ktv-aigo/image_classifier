import os
from preprocessing import processor
import dataset.dataloader as data_loader
from sklearn import preprocessing

path = "/home/ktranrb/Desktop/image_classifier/dataset/train"

le = preprocessing.LabelEncoder()

p = processor.Processor(28,28)

loader = data_loader.DataLoader(preprocessor=p)
data, label = loader.load(path)
le.fit(label)
labels = le.transform(label)
print(label[:5], labels[:5])