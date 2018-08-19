#!/usr/bin/env python

import time

import rospy
import roslib
import rospkg
import dlib

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import cv2
import os
import pickle
import sys

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

from person_detector.msg import faceDetails
from person_detector.srv import *

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir,'..','models')
dlibModelDir = os.path.join(modelDir,'dlib')
openFaceModelDir = os.path.join(modelDir,'openface','openface')
classifierModelDir = os.path.join(modelDir,'openface')

dModel = 'shape_predictor_68_face_landmarks.dat'
nModel = 'nn4.small2.v1.t7'
classifier = 'classifier.pkl'

dlibFacePredictor = os.path.join(dlibModelDir,dModel)
networkModel = os.path.join(openFaceModelDir,nModel)
classifierModel = os.path.join(classifierModelDir,classifier)
width = 320/2
height = 240/2
imgDim = 96
cuda = False
threshold = 0.7
verbose = False
font = cv2.FONT_HERSHEY_DUPLEX

rospy.init_node("FaceRecognition_server")

if rospy.has_param(rospy.get_name() + '/rec_tol'):
  threshold = float(rospy.get_param(rospy.get_name() + '/rec_tol'))

with open(classifierModel,'r') as f:
  if sys.version_info[0] < 3:
    (le, clf) = pickle.load(f) # le - label and clf - classifier
  else:
    (le, clf) = pickle.load(f, encoding='latin1') # le - label and clf - classifier

align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(
  networkModel,
  imgDim=imgDim,
  cuda=cuda)

cv_image = []

# Set threshold from command line
#ap = argparse.ArgumentParser()
#ap.add_argument("-t","--threshold", required=False, help="Threshold for recognizer (%)")
#args = vars(ap.parse_args())

#if args['threshold'] is not None:
#  threshold = args['threshold']

rospy.loginfo("[FaceRecognition_server] Enabled w/ threshold of %.2f"%threshold)

def recognize(reps, locations):
  persons = []
  confidences = []
  
  for i in range(len(reps)):
      rep = reps[i]
      location = locations[i]

      try:
          rep = rep.reshape(1, -1)
      except:
          print ("No Face detected")
          return (None, None)

      start = time.time()
      predictions = clf.predict_proba(rep).ravel()

      maxI = np.argmax(predictions)
      #max2 = np.argsort(predictions)[-3:][::-1][1] # Get second prediction

      persons.append(le.inverse_transform(maxI))
      #print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2])) # Print second prediction
      confidences.append(predictions[maxI])
    
      if isinstance(clf, GMM):
          dist = np.linalg.norm(rep - clf.means_[maxI])
          print("  + Distance from the mean: {}".format(dist))
          pass
  return (persons, confidences, locations)

def imageCb(data, rects):
  try:
    cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")
  except CvBridgeError as e:
    print e

  # Run detection
  if cv_image is None:
    raise Exception("Unable to load image/frame")

  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

  # Get all bounding boxes
  bb = dlib.rectangles()
  if len(rects) == 0:
    # bb = align.getLargestFaceBoundingBox(cv_image) # Get largest face bounding box
    bb = align.getAllFaceBoundingBoxes(cv_image)
  else:
    for i in range(len(rects)):
      bb.append(dlib.rectangle(rects[i].l, rects[i].t, rects[i].r, rects[i].b))
   
  if bb is None:
    return (None,None)

  alignedFaces = []
  for box in bb:
    alignedFaces.append(
      align.align(
        imgDim,
        cv_image,
        box,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

  if alignedFaces is None:
    raise Exception("Unable to align the frame")

  reps = []
  for alignedFace in alignedFaces:
      reps.append(net.forward(alignedFace))

  (persons, confidences, locations) = recognize(reps, bb) 

  resp = face_recognitionResponse()

  if len(rects) == 0:
    for i in range(len(persons)):
      person = faceDetails()

      person.name = persons[i]
      person.confidence = confidences[i]

      if confidences[i] <= threshold:
        person.name = "Unknown"  

      person.left = locations[i].left()
      person.right = locations[i].right()
      person.top = locations[i].top()
      person.bottom = locations[i].bottom()

      resp.person.append(person)
  else:
    for i in range(len(rects)):
      person = faceDetails()

      person.name = persons[i]
      person.confidence = confidences[i]

      if confidences[i] <= threshold:
        person.name = "Unknown"

      resp.person.append(person)

  return resp

def handle_FR(req):
  resp = face_recognitionResponse()

  try:
    resp = imageCb(req.im, req.rects)
  except rospy.ROSInterruptException:
    resp.person = faceDetails()
    # print("Could not recognize people")
    rospy.logerr("[FaceRecognition_server] ROS Interrupt!")
  return resp

def openFace_server():
  s = rospy.Service("FaceRecognition", face_recognition, handle_FR)
  #print("Open Face Recognition Server Enabled!")
  rospy.loginfo("[FaceRecognition_server] Open Face Recognition Server Enabled!")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  openFace_server()
