#!/usr/bin/python
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils import face_utils
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True,
                help="path to facial landmark predictors")
ap.add_argument("-i","--image",required=True,
                help="path to input image")
args = vars(ap.parse_args())

## Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

## Define a dictionary that maps the indexes of the facial landmarks to specific face regions
FACIAL_LANDMARK_IDXS = OrderedDict([
	("jaw", (0, 17)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("nose", (27, 35)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("mouth", (48, 68)),
])

def visualize_facial_landmarks(im, shape, colors=None, alpha=0.75):
  # Create two copies of the input image -- one for the overlay and one for the final output image
  overlay = image.copy()
  output = image.copy()
  
  # If the colors list is None, initialize it with a unique color for each facial landmark region
  if colors is None:
    colors = [
      (19, 199, 109),  (79, 76, 240),  (230, 159, 23),
			(168, 100, 168), (158, 163, 32), (163, 38, 32), 
			(180, 42, 220)
    ]
  
  # Loop over the facial landmark regions individually
  for (i, name) in enumerate(FACIAL_LANDMARK_IDXS.keys()):
    # Grab the (x,y)-coordinate associated with the face landmark
    (j,k) = FACIAL_LANDMARK_IDXS[name]
    pts = shape[j:k]
    
    # Check if you are supposed to draw the jawline
    if name == 'jaw':
      # since the jawline is a non-enclosed facial region, just draw lines between the (x,y)-coordinates
      for l in range(1, len(pts)):
        ptA = tuple(pts[l-1])
        ptB = tuple(pts[l])
        cv2.line(overlay, ptA, ptB, colors[i],2)
      
    # Otherwise, compute the convex hull of the facial landmark coordinates point and display it
    else:
      hull = cv2.convexHull(pts)
      cv2.drawContours(overlay, [hull], -1, colors[i], -1)
  
  # Overlay the overlay onto the original image
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
  
  # Return the output image
  return output
  
## Load the input image, resize it, and convert it to grayscale
#image = cv2.imread(args['image'])

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

image = frame

image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## Detect faces in the grayscale image
rects = detector(gray,1)

## Loop over the face detections
for (i, rect) in enumerate(rects):
  # Determine the facial landmarks for the face region, then convert the landmark (x,y)-coordinates
  # to the NumPy array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)
  
  # Loop over the face parts individually
  for (name, (i,j)) in FACIAL_LANDMARK_IDXS.items():
    # Clone the original image so we can draw on it, then display the name of the face
    # part on the image
    clone = image.copy()
    cv2.putText(clone, name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
  
    # Loop over the subset of the facial landmarks, drawing the specific face part
    for (x,y) in shape[i:j]:
      cv2.circle(clone, (x,y), 1, (0,0,255), -1)
      
      # Extract the ROI of the face region as a separate image
      (x, y, w, h) = cv2.boundingRect(np.array(shape[i:j]))
      roi = image[y:y + h, x:x + w]
      roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
      
      # Show the particular face part
      cv2.imshow("ROI", roi)
      cv2.imshow("Image", clone)
      cv2.waitKey(0)
      
    # Visualize all facial landmarks with a transparent overlay
    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("Image",output)
    cv2.waitKey(0)
