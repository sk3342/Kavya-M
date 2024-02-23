import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Image
image = cv2.imread('your_image.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Face Detection
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Store face data in a pandas DataFrame
face_data = pd.DataFrame(columns=['x', 'y', 'width', 'height'])
for (x, y, w, h) in faces:
    face_data = face_data.append({'x': x, 'y': y, 'width': w, 'height': h}, ignore_index=True)

# Display Result
for index, row in face_data.iterrows():
    x, y, w, h = row['x'], row['y'], row['width'], row['height']
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Save the result image
cv2.imwrite('result_image.jpg', image)

# Display the result image
cv2.imshow('Face Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print face data
print("Detected Faces:")
print(face_data)
 

#ijmport libsrariewsa
#import file using which oulf be gray scale
#print image
#%matplotlib iinline
#convert image of multiple faces in grascalea d use plt .show 
#classifier and dettect multiscale and n = len(faces) face foudn or not logic
#after detect face , annotatae the face
#pip install mtcnn
#matplotlib.paches import rectangle, circle mtcnn
#import time
