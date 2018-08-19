#!/usr/bin/env python

import time

import rospy
import rospkg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import os
import pickle
import sys

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

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

width = 90
height = 90
imgDim = 96
cuda = True
threshold = 0.5
verbose = False
font = cv2.FONT_HERSHEY_DUPLEX

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

print "Model ready"

def getRep(bgrImg):
  start = time.time()
  if bgrImg is None:
      raise Exception("Unable to load image/frame")

  rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

  if verbose:
      print("  + Original size: {}".format(rgbImg.shape))
  if verbose:
      print("Loading the image took {} seconds.".format(time.time() - start))

  start = time.time()

  # Get the largest face bounding box
  # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

  # Get all bounding boxes
  bb = align.getAllFaceBoundingBoxes(rgbImg)

  if bb is None:
      # raise Exception("Unable to find a face: {}".format(imgPath))
      return None,None
  if verbose:
      print("Face detection took {} seconds.".format(time.time() - start))

  start = time.time()

  alignedFaces = []
  for box in bb:
      alignedFaces.append(
          align.align(
              imgDim,
              rgbImg,
              box,
              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

  if alignedFaces is None:
      raise Exception("Unable to align the frame")
  if verbose:
      print("Alignment took {} seconds.".format(time.time() - start))

  start = time.time()

  reps = []
  for alignedFace in alignedFaces:
      reps.append(net.forward(alignedFace))

  if verbose:
      print("Neural network forward pass took {} seconds.".format(
          time.time() - start))

  # print (reps)
  return reps,bb

def infer(img):
  reps, locations = getRep(img)
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
      #print (predictions)
      maxI = np.argmax(predictions)
      #max2 = np.argsort(predictions)[-3:][::-1][1]
      old_size = len(persons)
      persons.append(le.inverse_transform(maxI))
      #print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
      # ^ prints the second prediction
      confidences.append(predictions[maxI])

      new_size = len(persons)

      if (new_size > old_size):
        print("Predict {} @ {} with {:.2f} confidence.".format(persons[-1].decode('utf-8'), location, confidences[-1]))
      if verbose:
          print("Prediction took {} seconds.".format(time.time() - start))
          pass
      if isinstance(clf, GMM):
          dist = np.linalg.norm(rep - clf.means_[maxI])
          print("  + Distance from the mean: {}".format(dist))
          pass
  return (persons, confidences, locations)

def imageCb(data):
  start_time = time.time()
  try:
    cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")
  except CvBridgeError as e:
    print e

  (persons, confidences, locations) = infer(cv_image)

  for i in range(len(persons)):
    left = locations[i].left()
    right = locations[i].right()
    top = locations[i].top()
    bottom = locations[i].bottom()

    cv2.rectangle(cv_image,(left,top),(right,bottom),(0,255,0),3)
    text = "{} (Confidence: {})".format(persons[i],confidences[i])
    cv2.putText(cv_image, text, (left, top - 20), font, 1.0, (255,255,255),1)

  cv2.imshow("Image Window", cv_image)
  cv2.waitKey(1)
  
  print(time.time()-start_time)

if __name__ == "__main__":
  rospy.init_node('openface_node')
  image_sub = rospy.Subscriber("camera/rgb/image_raw",Image,imageCb, queue_size=1, buff_size=2**24)

  if verbose:
    print ("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
    start = time.time()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
