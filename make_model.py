import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os 
import dlib

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
# Initialize the classifier with conservative hyperparameters
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=100,
    learning_rate=0.1,
    reg_lambda=0.1,
    reg_alpha=0.0,
)


faces = []
labels = []
people = os.listdir('./face/')
for person in people:
    pics = os.listdir('./face/'+person)
    for direction in pics:
        # print('./face/'+person+'/'+ direction)
        gray_image = cv2.imread('./face/'+person+'/'+ direction,cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.resize(gray_image,(100,100))
        gray_image = gray_image.flatten()
        faces.append(gray_image)
        labels.append(person)
        





X_train = faces
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels)



my_dict = {'hassan': 0, 'soheyl': 1}

print(y_train)
# Train the model
# print(X_train[0])


model.fit(X_train, y_train)

gray_image = cv2.imread('./front.jpg' ,cv2.IMREAD_GRAYSCALE)
gray_image = cv2.resize(gray_image,(100,100))
gray_image = gray_image.flatten()




print(model.predict([gray_image]))

