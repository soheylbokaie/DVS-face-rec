import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Load and preprocess images
def preprocess_image(target_size=(100, 100)):
    faces_l = []
    labels = []
    image_folder = './static/imgs/'  # Replace with the path to your training image folder
    
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]
    for imagep in image_paths :    
        image = cv2.imread(imagep, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(image,1.7,1)
        
        for (x, y, w, h) in faces:
            face = image[ y:y+h,x:x+w]
            face = cv2.resize(face, target_size)
            face = face.flatten()  # Flatten to 1D array
            faces_l.append(face)
            labels.append(''.join(imagep.split('/')[-1].split('.')[0]))
    return faces_l,labels

# Load and preprocess your dataset
# Create lists to store images and corresponding labels


# Add images and labels to the lists
# For example:
# images.append(preprocess_image('path_to_image.jpg'))
# labels.append('person_name')

images,labels = preprocess_image()
# Convert lists to numpy arrays
X = images
y = labels
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a k-NN classifier
print(X_train)
k = 3  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X, y)

# Evaluate the classifier
# accuracy = knn_classifier.score(X_test, y_test)
print("Accuracy:")
import joblib

# Assuming you've trained the k-NN classifier and stored it in the variable knn_classifier
model_filename = 'knn_model.pkl'
joblib.dump(knn_classifier, model_filename)