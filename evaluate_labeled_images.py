import cv2
import os
from os import listdir
from os.path import isfile, join
from sys import platform
import argparse
import math
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
import pyopenpose as op

import sys
import datetime

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

import csv

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import collections

import itertools

import time

def is_overlapping(rect1, rect2):
    x_overlap = False
    if (rect2[0] < rect1[0] < rect2[1]) or (rect2[0] < rect1[1] < rect2[1]):
        x_overlap = True
    if (rect1[0] < rect2[0] < rect1[1]) or (rect1[0] < rect2[1] < rect1[1]):
        x_overlap = True

    y_overlap = False
    if (rect2[2] < rect1[2] < rect2[3]) or (rect2[2] < rect1[3] < rect2[3]):
        y_overlap = True
    if (rect1[2] < rect2[2] < rect1[3]) or (rect1[2] < rect2[3] < rect1[3]):
        y_overlap = True

    if x_overlap and y_overlap:
        return True
    else:
        return False

#return the camera's focal length in terms of mm
def get_focal_length(filename, base_folder):
    image = Image.open(base_folder + filename)
    exifdata = image.getexif()

    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)

        if tag == 'FocalLength':
            focalLength = data.numerator

            return focalLength

    return 35 #default focal length if we cant find it on metadata

#takes as an input the pixel's X location and converts it into mm x location on sensor
def pixelXtoSensorX(sensorWidth, imageWidth, pixelX):
    return (pixelX * sensorWidth) / imageWidth

#takes as an input the pixel's Y location and converts it into mm y location on sensor
def pixelYtoSensorY(sensorHeight, imageHeight, pixelY):
    return (pixelY * sensorHeight) / imageHeight

def distanceBetweenTwoPoints(pos1 = [], pos2 = []):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

#calculates the real world 3D locations of the people by using the pupil keypoints of OpenPose
def pupil_distance(upperBodylessKeypoints, rect_index, sensorWidth, sensorHeight, imageWidth, imageHeight, focalLength,
                   realPupilDistance):
    faceLocations = []

    eyeMidPointPixelLocations = []

    pIndexes = []

    pIndex = 0
    for upperBodylessKeypoint in upperBodylessKeypoints:

        if upperBodylessKeypoint[15][2] != 0 and upperBodylessKeypoint[16][2] != 0 and upperBodylessKeypoint[16][0] > upperBodylessKeypoint[15][0]:
            eye1Xlocation = upperBodylessKeypoint[15][0]
            eye1Ylocation = upperBodylessKeypoint[15][1]

            eye2Xlocation = upperBodylessKeypoint[16][0]
            eye2Ylocation = upperBodylessKeypoint[16][1]

            eye1Xmm = pixelXtoSensorX(sensorWidth, imageWidth, eye1Xlocation)
            eye1Ymm = pixelYtoSensorY(sensorHeight, imageHeight, eye1Ylocation)

            eye2Xmm = pixelXtoSensorX(sensorWidth, imageWidth, eye2Xlocation)
            eye2Ymm = pixelYtoSensorY(sensorHeight, imageHeight, eye2Ylocation)

            pupillaryDistance = math.sqrt((eye1Xmm - eye2Xmm) ** 2 + (eye1Ymm - eye2Ymm) ** 2)

            #if the person is looking sideways to the camera and we are detecting only 1 of the ears, fix the pupillary distance
            if (upperBodylessKeypoint[17][2] == 0 and upperBodylessKeypoint[18][2] != 0) or (upperBodylessKeypoint[17][2] != 0 and upperBodylessKeypoint[18][2] == 0):
                if upperBodylessKeypoint[18][2] != 0:
                    ear2Xlocation = upperBodylessKeypoint[18][0]
                    ear2Ylocation = upperBodylessKeypoint[18][1]

                    ear2Xmm = pixelXtoSensorX(sensorWidth, imageWidth, ear2Xlocation)
                    ear2Ymm = pixelYtoSensorY(sensorHeight, imageHeight, ear2Ylocation)

                    pupillaryDistance = math.sqrt((eye1Xmm - ear2Xmm) ** 2 + (eye1Ymm - ear2Ymm) ** 2)
                    pupillaryDistance = pupillaryDistance / 1.95

                if upperBodylessKeypoint[17][2] != 0:
                    ear1Xlocation = upperBodylessKeypoint[17][0]
                    ear1Ylocation = upperBodylessKeypoint[17][1]

                    ear1Xmm = pixelXtoSensorX(sensorWidth, imageWidth, ear1Xlocation)
                    ear1Ymm = pixelYtoSensorY(sensorHeight, imageHeight, ear1Ylocation)

                    pupillaryDistance = math.sqrt((eye2Xmm - ear1Xmm) ** 2 + (eye2Ymm - ear1Ymm) ** 2)
                    pupillaryDistance = pupillaryDistance / 1.95


            # camera focal point is 0,0,0
            # these positions are the middle of the eyes on the sensor in terms of mm
            eyeSensorPointXmm = (eye1Xmm + eye2Xmm) / 2
            eyeSensorPointYmm = (eye1Ymm + eye2Ymm) / 2
            eyeSensorPointXmm = (-1) * (eyeSensorPointXmm - (sensorWidth / 2))
            eyeSensorPointYmm = (eyeSensorPointYmm - (sensorHeight / 2))
            eyeSensorPointZmm = focalLength

            cameraToFaceZDistance = 0

            if pupillaryDistance != 0:
                cameraToFaceZDistance = (focalLength * realPupilDistance) / pupillaryDistance

            eyeRealPointXmm = (-1) * (cameraToFaceZDistance / focalLength) * eyeSensorPointXmm
            eyeRealPointYmm = (-1) * (cameraToFaceZDistance / focalLength) * eyeSensorPointYmm
            eyeRealPointZmm = cameraToFaceZDistance

            faceLocations.append([eyeRealPointXmm, eyeRealPointYmm, eyeRealPointZmm])

            eyeMidPointPixelLocations.append([(eye1Xlocation+eye2Xlocation)/2, (eye1Ylocation+eye2Ylocation)/2])

            pIndexes.append(str(pIndex) + '-' + str(rect_index))

            pIndex += 1
        else:
            pIndex += 1
            continue
    return pIndexes, eyeMidPointPixelLocations, faceLocations

#calculates the real world 3D locations of the people by using the shoulder keypoints of OpenPose
def shoulder_distance(upperBodylessKeypoints, rect_index, sensorWidth, sensorHeight, imageWidth, imageHeight,
                      focalLength, realShoulderWidth):
    shoulderLocations = []

    shoulderMidPointPixelLocations = []

    pIndexes = []

    pIndex = 0
    for upperBodylessKeypoint in upperBodylessKeypoints:

        if upperBodylessKeypoint[2][2] != 0 and upperBodylessKeypoint[5][2] != 0:

            if upperBodylessKeypoint[1][2] != 0:
                shoulder1length = math.sqrt((upperBodylessKeypoint[2][0] - upperBodylessKeypoint[1][0])**2 + (upperBodylessKeypoint[2][1] - upperBodylessKeypoint[1][1])**2)
                shoulder2length = math.sqrt((upperBodylessKeypoint[1][0] - upperBodylessKeypoint[5][0]) ** 2 + (upperBodylessKeypoint[1][1] - upperBodylessKeypoint[5][1]) ** 2)

                if (shoulder1length / shoulder2length) > 1.25 or (shoulder2length / shoulder1length) > 1.25:
                    continue

            shoulder1Xlocation = upperBodylessKeypoint[2][0]
            shoulder1Ylocation = upperBodylessKeypoint[2][1]

            shoulder2Xlocation = upperBodylessKeypoint[5][0]
            shoulder2Ylocation = upperBodylessKeypoint[5][1]

            shoulder1Xmm = pixelXtoSensorX(sensorWidth, imageWidth, shoulder1Xlocation)
            shoulder1Ymm = pixelYtoSensorY(sensorHeight, imageHeight, shoulder1Ylocation)

            shoulder2Xmm = pixelXtoSensorX(sensorWidth, imageWidth, shoulder2Xlocation)
            shoulder2Ymm = pixelYtoSensorY(sensorHeight, imageHeight, shoulder2Ylocation)

            shoulderDistance = math.sqrt((shoulder1Xmm - shoulder2Xmm) ** 2 + (shoulder1Ymm - shoulder2Ymm) ** 2)

            # camera focal point is 0,0,0
            # these positions are the middle of the shoulders on the sensor in terms of mm
            shoulderSensorPointXmm = (shoulder1Xmm + shoulder2Xmm) / 2
            shoulderSensorPointYmm = (shoulder1Ymm + shoulder2Ymm) / 2
            shoulderSensorPointXmm = (-1) * (shoulderSensorPointXmm - (sensorWidth / 2))
            shoulderSensorPointYmm = (shoulderSensorPointYmm - (sensorHeight / 2))
            shoulderSensorPointZmm = focalLength

            cameraToShoulderZDistance = 0

            if shoulderDistance != 0:
                cameraToShoulderZDistance = (focalLength * realShoulderWidth) / shoulderDistance

            shoulderRealPointXmm = (-1) * (cameraToShoulderZDistance / focalLength) * shoulderSensorPointXmm
            shoulderRealPointYmm = (-1) * (cameraToShoulderZDistance / focalLength) * shoulderSensorPointYmm
            shoulderRealPointZmm = cameraToShoulderZDistance

            shoulderLocations.append([shoulderRealPointXmm, shoulderRealPointYmm, shoulderRealPointZmm])

            shoulderMidPointPixelLocations.append([(shoulder1Xlocation+shoulder2Xlocation)/2, (shoulder1Ylocation+shoulder2Ylocation)/2])

            pIndexes.append(str(pIndex) + '-' + str(rect_index))
            pIndex += 1

        else:
            pIndex += 1
            continue
    return pIndexes, shoulderMidPointPixelLocations, shoulderLocations

#calculates the real world 3D locations of the people by using the torso keypoints of OpenPose
def body_distance(eyelessKeypoints, rect_index, sensorWidth, sensorHeight, imageWidth, imageHeight, focalLength,
                  realUpperBodyLength):
    bodyLocations = []

    bodyMidPointPixelLocations = []

    pIndexes = []

    pIndex = 0
    for eyelessKeypoint in eyelessKeypoints:

        if eyelessKeypoint[1][2] != 0 and eyelessKeypoint[8][2] != 0:
            neckXlocation = eyelessKeypoint[1][0]
            neckYlocation = eyelessKeypoint[1][1]

            midhipXlocation = eyelessKeypoint[8][0]
            midhipYlocation = eyelessKeypoint[8][1]

            neckXmm = pixelXtoSensorX(sensorWidth, imageWidth, neckXlocation)
            neckYmm = pixelYtoSensorY(sensorHeight, imageHeight, neckYlocation)

            midhipXmm = pixelXtoSensorX(sensorWidth, imageWidth, midhipXlocation)
            midhipYmm = pixelYtoSensorY(sensorHeight, imageHeight, midhipYlocation)

            bodyDistance = math.sqrt((neckXmm - midhipXmm) ** 2 + (neckYmm - midhipYmm) ** 2)

            # camera focal point is 0,0,0
            # these positions are the middle of the body on the sensor in terms of mm
            bodySensorPointXmm = (neckXmm + midhipXmm) / 2
            bodySensorPointYmm = (neckYmm + midhipYmm) / 2
            bodySensorPointXmm = (-1) * (bodySensorPointXmm - (sensorWidth / 2))
            bodySensorPointYmm = (bodySensorPointYmm - (sensorHeight / 2))
            bodySensorPointZmm = focalLength

            cameraToBodyZDistance = 0

            if bodyDistance != 0:
                cameraToBodyZDistance = (focalLength * realUpperBodyLength) / bodyDistance

            bodyRealPointXmm = (-1) * (cameraToBodyZDistance / focalLength) * bodySensorPointXmm
            bodyRealPointYmm = (-1) * (cameraToBodyZDistance / focalLength) * bodySensorPointYmm
            bodyRealPointZmm = cameraToBodyZDistance

            bodyLocations.append([bodyRealPointXmm, bodyRealPointYmm, bodyRealPointZmm])

            bodyMidPointPixelLocations.append([(neckXlocation + midhipXlocation) / 2, (neckYlocation + midhipYlocation) / 2])

            pIndexes.append(str(pIndex) + '-' + str(rect_index))
            pIndex += 1

        else:
            pIndex += 1
            continue
    return pIndexes, bodyMidPointPixelLocations, bodyLocations

#returns the number of people that has both of the torso keypoints of OpenPose
def upperBodyCount(keypoints):
    upperBodyCount = 0
    t = 0
    while t < keypoints.shape[0]:
        if keypoints[t, 1, 2] != 0 and keypoints[t, 8, 2] != 0 and (keypoints[t, 9, 2] != 0 or keypoints[t, 12, 2] != 0):
            upperBodyCount = upperBodyCount + 1
        t = t + 1
    return upperBodyCount

#returns the number of people that has both of the pupil keypoints of OpenPose
def eyePairCount(keypoints):
    eyePairs = 0
    t = 0
    while t < keypoints.shape[0]:
        if keypoints[t, 15, 2] != 0 and keypoints[t, 16, 2] != 0 and keypoints[t, 16, 0] > keypoints[t, 15, 0]:
            eyePairs = eyePairs + 1
        t = t + 1
    return eyePairs

#returns the keypoints without torso keypoints
def extractKeypointsWithoutUpperBodies(keypoints):
    upperBodylessKeypoints = []
    p = 0
    while p < keypoints.shape[0]:
        if not(keypoints[p, 1, 2] != 0 and keypoints[p, 8, 2] != 0 and (keypoints[p, 9, 2] != 0 or keypoints[p, 12, 2] != 0)):
            upperBodylessKeypoints.append(np.expand_dims(keypoints[p, :, :], axis=0))
        p = p + 1
    if len(upperBodylessKeypoints) > 0:
        upperBodylessKeypoints = np.concatenate(upperBodylessKeypoints, axis=0)
    return upperBodylessKeypoints

#returns the keypoints without pupil keypoints
def extractKeypointsWithoutEyePairs(keypoints):
    eyelessKeypoints = []
    p = 0
    while p < keypoints.shape[0]:
        if not(keypoints[p, 15, 2] != 0 and keypoints[p, 16, 2] != 0) or not(keypoints[p, 15, 0] < keypoints[p, 16, 0]):
            eyelessKeypoints.append(np.expand_dims(keypoints[p, :, :], axis=0))
        p = p + 1
    if len(eyelessKeypoints) > 0:
        eyelessKeypoints = np.concatenate(eyelessKeypoints, axis=0)
    return eyelessKeypoints

#takes as an input the rectangle groups and person_locations
#gives the startX,endX,startY,endY locations of the merged groups
def give_group_locations(rect_groups, person_locations):
    merged_rect_locations = []
    for rect_group in rect_groups:
        startXs = []
        endXs = []
        startYs = []
        endYs = []
        for person in rect_group:
            startXs.append(person_locations[person][0])
            endXs.append(person_locations[person][1])
            startYs.append(person_locations[person][2])
            endYs.append(person_locations[person][3])
        startX = min(startXs)
        endX = max(endXs)
        startY = min(startYs)
        endY = max(endYs)
        merged_rect_locations.append([startX, endX, startY, endY])
    return merged_rect_locations

def get_rectangle_lists(classes, confidences, boxes):
    # list of [startX, endX, startY, endY] of every person rectangle
    person_rectangles = []
    if len(classes) > 0:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            if classId != 0:
                continue
            left, top, width, height = box
            person_rectangles.append([left, left + width, top, top + height])

    person_rectangles = sorted(person_rectangles, key=itemgetter(0))
    rectangle_groups = [[] for i in range(len(person_rectangles))]
    if (len(person_rectangles) > 1):
        for i in range(len(person_rectangles)):
            for k in range(i + 1, len(person_rectangles)):
                if is_overlapping(person_rectangles[i], person_rectangles[k]):
                    rectangle_groups[i].append(k)

    for i in range(len(rectangle_groups)):
        for element1 in rectangle_groups[i]:
            for element2 in rectangle_groups[element1]:
                rectangle_groups[i].append(element2)

    for i in range(len(rectangle_groups)):
        rectangle_groups[i].append(i)

    for i in range(len(rectangle_groups)):
        rectangle_groups[i] = sorted(rectangle_groups[i])

    to_remove = []
    for i in range(len(rectangle_groups) - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            set1 = set(rectangle_groups[i])
            set2 = set(rectangle_groups[k])
            if set1.issubset(set2):
                if i not in to_remove:
                    to_remove.append(i)

    temp_rectangle_groups = []
    for i in range(len(rectangle_groups)):
        if i not in to_remove:
            temp_rectangle_groups.append(rectangle_groups[i])
    rectangle_groups = temp_rectangle_groups
    del temp_rectangle_groups

    # startX,endX,startY,endY locations of the merged rectangle groups
    merged_rect_locations = give_group_locations(rectangle_groups, person_rectangles)

    return person_rectangles, rectangle_groups, merged_rect_locations

#takes the 3D location estimations of the people that were made on pupil, shoulder and torso keypoints
#for each person, chooses the estimation that is the closest to the camera
#return a list that contains these chosen estimations
def minimum_locations(pupilLocations, pupilIndexes, shoulderLocations, shoulderIndexes, detectLocations, detectIndexes):

    locationsDictionary = dict()
    personIndexes = []
    for pIndex in pupilIndexes:
        if pIndex not in personIndexes:
            personIndexes.append(pIndex)
    for pIndex in shoulderIndexes:
        if pIndex not in personIndexes:
            personIndexes.append(pIndex)
    for pIndex in detectIndexes:
        if pIndex not in personIndexes:
            personIndexes.append(pIndex)

    for personIndex in personIndexes:
        locationsDictionary.update({personIndex: []})

    i = 0
    for location in pupilLocations:
        locationsDictionary[pupilIndexes[i]].append(location)
        i += 1

    i = 0
    for location in shoulderLocations:
        locationsDictionary[shoulderIndexes[i]].append(location)
        i += 1

    i = 0
    for location in detectLocations:
        locationsDictionary[detectIndexes[i]].append(location)
        i += 1

    minimumLocations = []
    for key in locationsDictionary:
        minLoc = sorted(locationsDictionary[key], key=itemgetter(2))[0]
        minimumLocations.append(minLoc)

    return minimumLocations

def average(l):
    llen = len(l)
    def divide(x): return x / llen
    return list(map(divide, map(sum, zip(*l))))

def averageOfDistancePairs(locations):
    avgDist = 0
    distCount = 0
    if (len(locations) > 1):
        for i in range(len(locations)):
            for k in range(i + 1, len(locations)):
                avgDist += distanceBetweenTwoPoints(locations[i], locations[k])
                distCount += 1
    if distCount > 0:
        return avgDist / distCount
    return 0

def removeFaultyKeypoints(keypoints, eyePairs, YOLO_person_count):

    correctedKeypoints = []

    areaIndexDict = dict()

    index = 0
    for keypoint in keypoints:
        area = areaOfKeypoints(keypoint)

        areaIndexDict.update({index: area})

        index += 1

    correctedAreaIndexes = collections.Counter(areaIndexDict).most_common(YOLO_person_count)
    correctedIndexes = []
    for correctedAreaIndex in correctedAreaIndexes:
        correctedIndex = correctedAreaIndex[0]
        correctedIndexes.append(correctedIndex)

    for correctedIndex in correctedIndexes:
        correctedKeypoints.append(keypoints[correctedIndex])

    correctedKeypoints = np.array(correctedKeypoints)

    return correctedKeypoints

def areaOfKeypoints(keypoint):
    xmin = np.amin(keypoint[:, 0])
    xmax = np.amax(keypoint[:, 0])

    ymin = np.amin(keypoint[:, 1])
    ymax = np.amax(keypoint[:, 1])
    return (xmax-xmin) * (ymax-ymin)

#takes the 3D estimated locations and corresponding person tags of the people in the image
#calculates the percent estimation errors between each pair
#returns the average of these percent estimation errors
def pairwise_distance_evaluation(estimation_dictionary, photoshoot_id):

    ground_truth_locations = dict()

    with open('labels/ground_truth_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            info = row
            if info[0] == photoshoot_id:
                if info[1][0] == 'P':
                    ground_truth_locations.update({info[1]: [float(info[2]), float(info[3]), float(info[4])]})

    estimation_dictionary_new = dict()

    for key in estimation_dictionary:
        if estimation_dictionary[key] != [0, 0, 0]:
            estimation_dictionary_new.update({key: estimation_dictionary[key]})

    estimation_dictionary = estimation_dictionary_new

    groundTruthPairwiseDistanceDictionary = dict()
    ground_truth_location_key_list = list(ground_truth_locations)

    for i in range(len(ground_truth_location_key_list)):
        for k in range(i + 1, len(ground_truth_location_key_list)):
            tag1 = ground_truth_location_key_list[i]
            tag2 = ground_truth_location_key_list[k]

            tag1num = float(tag1.replace('P', ''))
            tag2num = float(tag2.replace('P', ''))

            if tag1num < tag2num:
                pair = tag1 + '-' + tag2
            else:
                pair = tag2 + '-' + tag1


            point1 = ground_truth_locations[tag1]
            point2 = ground_truth_locations[tag2]
            distance = distanceBetweenTwoPoints(point1, point2)
            groundTruthPairwiseDistanceDictionary.update({pair: distance})

    percent_errors = []

    estimation_key_list = list(estimation_dictionary)

    pair_wise_distance_estimation_dictionary = dict()

    for i in range(len(estimation_key_list)):
        for k in range(i + 1, len(estimation_key_list)):

            tag1 = estimation_key_list[i]
            tag2 = estimation_key_list[k]
            point1 = estimation_dictionary[tag1]
            point2 = estimation_dictionary[tag2]

            tag1num = float(tag1.replace('P', ''))
            tag2num = float(tag2.replace('P', ''))

            if tag1num < tag2num:
                pair = tag1 + '-' + tag2
            else:
                pair = tag2 + '-' + tag1

            estimated_distance = distanceBetweenTwoPoints(point1, point2)

            pair_wise_distance_estimation_dictionary.update({pair: estimated_distance})

    for key in pair_wise_distance_estimation_dictionary:
        percent_error = (abs(pair_wise_distance_estimation_dictionary[key] - groundTruthPairwiseDistanceDictionary[key]) / groundTruthPairwiseDistanceDictionary[key]) * 100
        percent_errors.append(percent_error)

    if len(percent_errors) == 0:
        return 0

    average_pairwise_percent_error = sum(percent_errors) / len(percent_errors)
    return average_pairwise_percent_error

# takes the pair-wise estimated distances as an input
# calculates the percent estimation errors between each pair
# returns the average of these percent estimation errors
def pairwise_distance_evaluation_2(pair_wise_distance_estimation_dictionary, photoshoot_id):
    ground_truth_locations = dict()

    with open('labels/ground_truth_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            info = row
            if info[0] == photoshoot_id:
                if info[1][0] == 'P':
                    ground_truth_locations.update({info[1]: [float(info[2]), float(info[3]), float(info[4])]})

    groundTruthPairwiseDistanceDictionary = dict()
    ground_truth_location_key_list = list(ground_truth_locations)

    for i in range(len(ground_truth_location_key_list)):
        for k in range(i + 1, len(ground_truth_location_key_list)):
            tag1 = ground_truth_location_key_list[i]
            tag2 = ground_truth_location_key_list[k]

            tag1num = float(tag1.replace('P', ''))
            tag2num = float(tag2.replace('P', ''))

            if tag1num < tag2num:
                pair = tag1 + '-' + tag2
            else:
                pair = tag2 + '-' + tag1

            point1 = ground_truth_locations[tag1]
            point2 = ground_truth_locations[tag2]
            distance = distanceBetweenTwoPoints(point1, point2)
            groundTruthPairwiseDistanceDictionary.update({pair: distance})

    percent_errors = []

    for key in pair_wise_distance_estimation_dictionary:
        percent_error = (abs(pair_wise_distance_estimation_dictionary[key] - groundTruthPairwiseDistanceDictionary[key]) / groundTruthPairwiseDistanceDictionary[key]) * 100
        percent_errors.append(percent_error)

    if len(percent_errors) == 0:
        return 0

    average_pairwise_percent_error = sum(percent_errors) / len(percent_errors)
    return average_pairwise_percent_error

#identifies which photoshoot a file belongs to
def photoshoot_identifier(filename):
    with open('labels/camera_locations_photoshoot_identifiers.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            info = row
            if info[2] == filename:
                return info[0]
    return -1

def averageOfList(inputList):
    if len(inputList) == 0:
        return 0
    return sum(inputList) / len(inputList)

# takes the 3D location estimations of each person and also the X, Y pixel locations of
# certain body parts of these people as input
# personLocations should be a list of lists where each sublist is a 3D location estimation
# ShoulderPixelPoints should be a list of lists where each sublist is the X, Y pixel location of keypoint 1
# on the 25 keypoint OpenPose model (middle of shoulders)
# PupilPixelPoints should be a list of lists where each sublist is the X, Y pixel location of the middle point
# of keypoints 15-16 on the 25 keypoint OpenPose model (middle of eyes)
# BodyPixelPoints should be a list of lists where each sublist is the X, Y pixel location of the middle point of
# keypoints 1-8 on the 25 keypoint OpenPose model (middle of torso)
# HeadPixelPoints should be a list of lists where each sublist is the X, Y pixel location of the center point of head
# filename should be the name of the file
# imageWidth and imageHeight should be the width and height of the final image
# for example, if 50% rescaling was applied, then imageWidth and imageHeight should be
# given as half of original dimensions
# the order of each list should be in the exact corresponding order of people
# for example, personLocations[n] would be a list of length 3 that is the 3D location estimation of the n'th person
# then, ShoulderPixelPoints[n], PupilPixelPoints[n], BodyPixelPoints[n] and HeadPixelPoints[n] must correspond
# to the body part pixel locations of the same n'th person
# for personLocations[n], at least one of ShoulderPixelPoints[n], PupilPixelPoints[n], BodyPixelPoints[n],
# HeadPixelPoints[n] must be provided
# this means that for any given person, at least one of the body part pixel location must be provided so that
# automatic matching can be done
# if any of these body pixel locations are missing, they should be given as [0,0]
# an example of a case where there are 2 estimated people in the picture would be:
# personLocations : [ [100,100,100] , [200,200,200] ]
# ShoulderPixelPoints : [ [0,0] , [0,0] ]
# PupilPixelPoints : [ [50,60] , [100,60] ]
# BodyPixelPoints : [ [50,160] , [100, 160] ]
# HeadPixelPoints : [ [0,0] , [0,0] ]
# this is an example of a case where the used system can detect the pupil and body pixel points but not the
# shoulder and head pixel points
# finally, the function returns the average of pair-wise percent distance estimation errors
def automatic_per_err(personLocations, ShoulderPixelPoints, PupilPixelPoints, BodyPixelPoints, HeadPixelPoints,
                      filename, imageWidth, imageHeight):

    #there must be at least 1 body pixel location provided for a given person location estimation
    #if all three body pixel locations are missing for a given person location, matching cannot be done
    for locIndex in range(len(personLocations)):
        if ShoulderPixelPoints[locIndex] == [0, 0] and PupilPixelPoints[locIndex] == [0, 0] and BodyPixelPoints[locIndex] == [0, 0] and HeadPixelPoints[locIndex] == [0, 0]:
            return 0

    closestPeopleTags = []
    closestDistances = []

    with open('labels/body_pixel_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        imageSpecificInformation = []
        for row in csv_reader:
            info = row
            if info[4] == filename:
                imageSpecificInformation.append(info)
        originalWidth = imageSpecificInformation[0][5]
        originalHeight = imageSpecificInformation[0][6]

        #if the image was rescaled, convert the body pixel points so that they correspond to the original dimensions
        for shoulderPixelPoint in ShoulderPixelPoints:
            shoulderPixelPoint[0] *= int(originalWidth) / imageWidth
            shoulderPixelPoint[1] *= int(originalHeight) / imageHeight

        for pupilPixelPoint in PupilPixelPoints:
            pupilPixelPoint[0] *= int(originalWidth) / imageWidth
            pupilPixelPoint[1] *= int(originalHeight) / imageHeight

        for bodyPixelPoint in BodyPixelPoints:
            bodyPixelPoint[0] *= int(originalWidth) / imageWidth
            bodyPixelPoint[1] *= int(originalHeight) / imageHeight

        for headPixelPoint in HeadPixelPoints:
            headPixelPoint[0] *= int(originalWidth) / imageWidth
            headPixelPoint[1] *= int(originalHeight) / imageHeight

        locIndex = 0
        for locIndex in range(len(personLocations)):
            shoulderPixelPoint = ShoulderPixelPoints[locIndex]
            pupilPixelPoint = PupilPixelPoints[locIndex]
            bodyPixelPoint = BodyPixelPoints[locIndex]
            headPixelPoint = HeadPixelPoints[locIndex]

            closestShoulderPoint = ['Person', 'Distance']
            closestPupilPoint = ['Person', 'Distance']
            closestBodyPoint = ['Person', 'Distance']
            closestHeadPoint = ['Person', 'Distance']

            shoulderMinDistance = 1000000
            pupilMinDistance = 1000000
            bodyMinDistance = 1000000
            headMinDistance = 1000000

            for imageSpecificInfo in imageSpecificInformation:
                if 'Shoulder' in imageSpecificInfo[1]:
                    if shoulderPixelPoint[0] != 0 and shoulderPixelPoint[1] != 0:
                        shoulderDistance = math.sqrt((shoulderPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (shoulderPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if shoulderDistance < shoulderMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestShoulderPoint[0] = personTag
                            closestShoulderPoint[1] = shoulderDistance
                            shoulderMinDistance = shoulderDistance

                if 'Eyes' in imageSpecificInfo[1]:
                    if pupilPixelPoint[0] != 0 and pupilPixelPoint[1] != 0:
                        pupilDistance = math.sqrt((pupilPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (pupilPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if pupilDistance < pupilMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestPupilPoint[0] = personTag
                            closestPupilPoint[1] = pupilDistance
                            pupilMinDistance = pupilDistance

                if 'Head' in imageSpecificInfo[1]:
                    if headPixelPoint[0] != 0 and headPixelPoint[1] != 0:
                        headDistance = math.sqrt((headPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (headPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if headDistance < headMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestHeadPoint[0] = personTag
                            closestHeadPoint[1] = headDistance
                            headMinDistance = headDistance

                if 'Torso' in imageSpecificInfo[1]:
                    if bodyPixelPoint[0] != 0 and bodyPixelPoint[1] != 0:
                        bodyDistance = math.sqrt((bodyPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (bodyPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if bodyDistance < bodyMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestBodyPoint[0] = personTag
                            closestBodyPoint[1] = bodyDistance
                            bodyMinDistance = bodyDistance

            closestPerson = 'Person'
            closestDistance = 1000000
            if closestShoulderPoint[1] != 'Distance':
                closestPerson = closestShoulderPoint[0]
                closestDistance = closestShoulderPoint[1]

            if closestPupilPoint[1] != 'Distance':
                if closestPupilPoint[1] < closestDistance:
                    closestPerson = closestPupilPoint[0]
                    closestDistance = closestPupilPoint[1]

            if closestBodyPoint[1] != 'Distance':
                if closestBodyPoint[1] < closestDistance:
                    closestPerson = closestBodyPoint[0]
                    closestDistance = closestBodyPoint[1]

            if closestHeadPoint[1] != 'Distance':
                if closestHeadPoint[1] < closestDistance:
                    closestPerson = closestHeadPoint[0]
                    closestDistance = closestHeadPoint[1]

            closestPeopleTags.append(closestPerson)
            closestDistances.append(closestDistance)

    #the closest people list must not have any repeating entries
    #if multiple location estimations are matched with the same person, then the match with the closest distance is
    #chosen

    indexesToRemove = []
    if len(closestPeopleTags) != len(set(closestPeopleTags)):
        repeatingEntries = set([x for x in closestPeopleTags if closestPeopleTags.count(x) > 1])
        for repeatingEntry in repeatingEntries:
            minDistance = 100000000
            indexToKeep = -1
            for i in range(len(closestPeopleTags)):
                if repeatingEntry == closestPeopleTags[i]:
                    if closestDistances[i] < minDistance:
                        minDistance = closestDistances[i]
                        indexToKeep = i

            for i in range(len(closestPeopleTags)):
                if repeatingEntry == closestPeopleTags[i]:
                    if i != indexToKeep:
                        indexesToRemove.append(i)

        for indexToRemove in indexesToRemove:
            closestPeopleTags[indexToRemove] = 'REMOVE'
            closestDistances[indexToRemove] = -1

        closestPeopleTags = list(filter(('REMOVE').__ne__, closestPeopleTags))
        closestDistances = list(filter((-1).__ne__, closestPeopleTags))

        if len(closestPeopleTags) != len(set(closestPeopleTags)):
            return 0, 0, 0, 0, 0, 0

    estimatedLocationsDictionary = dict()

    for closestPersonTagIndex in range(len(closestPeopleTags)):
        estimatedLocationsDictionary.update({closestPeopleTags[closestPersonTagIndex]: personLocations[closestPersonTagIndex]});

    photoshoot_id = photoshoot_identifier(filename)

    if photoshoot_id == -1:
        return 0

    per_err = pairwise_distance_evaluation(estimatedLocationsDictionary, photoshoot_id)

    return per_err


# same as automatic_per_err but instead of 3D locations, the function takes the number of people
# and the estimated distances between them as inputs
# if there are n people in the picture, they would be indexed 0,1,2,3,...,n for this function
# socialDistances should be a list of all the pair-wise estimated distances
# the order of the distances should be in the following order of human pairs:
# 0-1, 0-2, ..., 0-n, 1-2, 1-3, ..., 1-n, 2-3, 2-4, ..., 2-n, ..., (n-1)-n
# the 4 body pixel point arrays should contain the respective body pixel locations of the people in the following human
# index order: 0,1,2,3,...,n
def automatic_per_err_2(numberOfPeople, socialDistances, ShoulderPixelPoints, PupilPixelPoints, BodyPixelPoints,
                        HeadPixelPoints, filename, imageWidth, imageHeight):

    if numberOfPeople < 2:
        return 0

    if math.comb(numberOfPeople, 2) != len(socialDistances):
        return 0

    #there must be at least 1 body pixel location provided for a person
    #if all three body pixel locations are missing for a given person location, matching cannot be done
    for locIndex in range(numberOfPeople):
        if ShoulderPixelPoints[locIndex] == [0, 0] and PupilPixelPoints[locIndex] == [0, 0] and BodyPixelPoints[locIndex] == [0, 0] and HeadPixelPoints[locIndex] == [0, 0]:
            return 0

    closestPeopleTags = []
    closestDistances = []

    with open('labels/body_pixel_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        imageSpecificInformation = []
        for row in csv_reader:
            info = row
            if info[4] == filename:
                imageSpecificInformation.append(info)
        originalWidth = imageSpecificInformation[0][5]
        originalHeight = imageSpecificInformation[0][6]

        #if the image was rescaled, convert the body pixel points so that they correspond to the original dimensions
        for shoulderPixelPoint in ShoulderPixelPoints:
            shoulderPixelPoint[0] *= int(originalWidth) / imageWidth
            shoulderPixelPoint[1] *= int(originalHeight) / imageHeight

        for pupilPixelPoint in PupilPixelPoints:
            pupilPixelPoint[0] *= int(originalWidth) / imageWidth
            pupilPixelPoint[1] *= int(originalHeight) / imageHeight

        for bodyPixelPoint in BodyPixelPoints:
            bodyPixelPoint[0] *= int(originalWidth) / imageWidth
            bodyPixelPoint[1] *= int(originalHeight) / imageHeight

        for headPixelPoint in HeadPixelPoints:
            headPixelPoint[0] *= int(originalWidth) / imageWidth
            headPixelPoint[1] *= int(originalHeight) / imageHeight

        locIndex = 0
        for locIndex in range(numberOfPeople):
            shoulderPixelPoint = ShoulderPixelPoints[locIndex]
            pupilPixelPoint = PupilPixelPoints[locIndex]
            bodyPixelPoint = BodyPixelPoints[locIndex]
            headPixelPoint = HeadPixelPoints[locIndex]

            closestShoulderPoint = ['Person', 'Distance']
            closestPupilPoint = ['Person', 'Distance']
            closestBodyPoint = ['Person', 'Distance']
            closestHeadPoint = ['Person', 'Distance']

            shoulderMinDistance = 1000000
            pupilMinDistance = 1000000
            bodyMinDistance = 1000000
            headMinDistance = 1000000

            for imageSpecificInfo in imageSpecificInformation:
                if 'Shoulder' in imageSpecificInfo[1]:
                    if shoulderPixelPoint[0] != 0 and shoulderPixelPoint[1] != 0:
                        shoulderDistance = math.sqrt((shoulderPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (shoulderPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if shoulderDistance < shoulderMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestShoulderPoint[0] = personTag
                            closestShoulderPoint[1] = shoulderDistance
                            shoulderMinDistance = shoulderDistance

                if 'Eyes' in imageSpecificInfo[1]:
                    if pupilPixelPoint[0] != 0 and pupilPixelPoint[1] != 0:
                        pupilDistance = math.sqrt((pupilPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (pupilPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if pupilDistance < pupilMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestPupilPoint[0] = personTag
                            closestPupilPoint[1] = pupilDistance
                            pupilMinDistance = pupilDistance

                if 'Head' in imageSpecificInfo[1]:
                    if headPixelPoint[0] != 0 and headPixelPoint[1] != 0:
                        headDistance = math.sqrt((headPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (headPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if headDistance < headMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestHeadPoint[0] = personTag
                            closestHeadPoint[1] = headDistance
                            headMinDistance = headDistance

                if 'Torso' in imageSpecificInfo[1]:
                    if bodyPixelPoint[0] != 0 and bodyPixelPoint[1] != 0:
                        bodyDistance = math.sqrt((bodyPixelPoint[0] - int(imageSpecificInfo[2]))**2 + (bodyPixelPoint[1] - int(imageSpecificInfo[3]))**2)
                        if bodyDistance < bodyMinDistance:
                            personTag = imageSpecificInfo[0]
                            closestBodyPoint[0] = personTag
                            closestBodyPoint[1] = bodyDistance
                            bodyMinDistance = bodyDistance

            closestPerson = 'Person'
            closestDistance = 1000000
            if closestShoulderPoint[1] != 'Distance':
                closestPerson = closestShoulderPoint[0]
                closestDistance = closestShoulderPoint[1]

            if closestPupilPoint[1] != 'Distance':
                if closestPupilPoint[1] < closestDistance:
                    closestPerson = closestPupilPoint[0]
                    closestDistance = closestPupilPoint[1]

            if closestBodyPoint[1] != 'Distance':
                if closestBodyPoint[1] < closestDistance:
                    closestPerson = closestBodyPoint[0]
                    closestDistance = closestBodyPoint[1]

            if closestHeadPoint[1] != 'Distance':
                if closestHeadPoint[1] < closestDistance:
                    closestPerson = closestHeadPoint[0]
                    closestDistance = closestHeadPoint[1]

            closestPeopleTags.append(closestPerson)
            closestDistances.append(closestDistance)

    #the closest people list must not have any repeating entries
    #if multiple location estimations are matched with the same person, then the match with the closest distance is
    #chosen
    indexesToRemove = []
    if len(closestPeopleTags) != len(set(closestPeopleTags)):
        repeatingEntries = set([x for x in closestPeopleTags if closestPeopleTags.count(x) > 1])
        for repeatingEntry in repeatingEntries:
            minDistance = 100000000
            indexToKeep = -1
            for i in range(len(closestPeopleTags)):
                if repeatingEntry == closestPeopleTags[i]:
                    if closestDistances[i] < minDistance:
                        minDistance = closestDistances[i]
                        indexToKeep = i

            for i in range(len(closestPeopleTags)):
                if repeatingEntry == closestPeopleTags[i]:
                    if i != indexToKeep:
                        indexesToRemove.append(i)

        for indexToRemove in indexesToRemove:
            closestPeopleTags[indexToRemove] = 'REMOVE'
            closestDistances[indexToRemove] = -1

        closestPeopleTags = list(filter(('REMOVE').__ne__, closestPeopleTags))
        closestDistances = list(filter((-1).__ne__, closestPeopleTags))

        if len(closestPeopleTags) != len(set(closestPeopleTags)):
            return 0

    pair_wise_dist_est_dict = dict()

    pair_index = 0
    for i in range(len(closestPeopleTags)):
        for k in range(i + 1, len(closestPeopleTags)):
            tag1 = float(closestPeopleTags[i].replace('P', ''))
            tag2 = float(closestPeopleTags[k].replace('P', ''))

            if tag1 < tag2:
                pair = closestPeopleTags[i] + '-' + closestPeopleTags[k]
            else:
                pair = closestPeopleTags[k] + '-' + closestPeopleTags[i]

            estimatedDistance = socialDistances[pair_index]
            pair_wise_dist_est_dict.update({pair: estimatedDistance})
            pair_index += 1

    photoshoot_id = photoshoot_identifier(filename)

    if photoshoot_id == -1:
        return 0

    per_err = pairwise_distance_evaluation_2(pair_wise_dist_est_dict, photoshoot_id)
    return per_err



#takes the 3D location estimations of each person and the filename as input
#returns the person detection rate
#for example if there are 10 people in the picture but 8 locations are provided, the function returns 0.8
def detection_rate(detectedPersonCount, filename):
    detected_person_count = detectedPersonCount

    taggedPeople = []

    with open('labels/body_pixel_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            info = row
            if info[4] == filename:
                personTag = info[0]
                if personTag not in taggedPeople:
                    taggedPeople.append(personTag)

    tagged_person_count = len(taggedPeople)

    if detected_person_count > tagged_person_count:
        return 1

    return detected_person_count / tagged_person_count

def false_discovery_rate(detectedPersonCount, filename):
    detected_person_count = detectedPersonCount

    taggedPeople = []

    with open('labels/body_pixel_locations.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            info = row
            if info[4] == filename:
                personTag = info[0]
                if personTag not in taggedPeople:
                    taggedPeople.append(personTag)

    tagged_person_count = len(taggedPeople)

    if detected_person_count < tagged_person_count:
        return 0

    falsePositives = detected_person_count - tagged_person_count
    return falsePositives / detected_person_count


def main():
    net = cv2.dnn_DetectionModel('yolo_models/yolov4.cfg.txt', 'yolo_models/yolov4.weights')
    net.setInputSize(704, 704)
    net.setInputScale(1.0/255)
    net.setInputSwapRB(True)

    # base_folder = 'C:/Users/tkmese/Documents/examples/paperimages/allphotos/'
    # sensorWidth = 36  # mms
    # sensorHeight = 24  # mms
    # scale_percent = 50 # downscale images by 50%
    # safeDistance = 2000 # mms

    #user input parameters
    base_folder = sys.argv[1]
    sensorWidth = float(sys.argv[2])
    sensorHeight = float(sys.argv[3])
    scale_percent = float(sys.argv[4])
    safeDistance = float(sys.argv[5])

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]

    params = dict()
    #params["model_folder"] = "../../../models/"
    params["model_folder"] = "openpose_models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    filenames = []
    detectedPersonCounts = []
    distancesStrings = []
    per_errors = []
    avg_distances = []
    violation_counts = []
    detection_rates = []
    false_discovery_rates = []

    realShoulderWidth = 38.9 * 10  # mms
    realEyeDistance = 6.3 * 10  # mms
    realUpperBodyLength = 44.45 * 10  # mms

    for filename in file_list:

        distances = []

        frame = cv2.imread(base_folder + filename)

        image = Image.open(base_folder + filename)

        focalLength = get_focal_length(filename, base_folder)
        if len(sys.argv) == 7:
            focalLength = float(sys.argv[6])

        with open('yolo_models/coco.names.txt', 'rt') as f:
            names = f.read().rstrip('\n').split('\n')

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, cv2.INTER_AREA)
        classes, confidences, boxes = net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

        person_rectangles, rectangle_groups, merged_rect_locations = get_rectangle_lists(classes, confidences, boxes)

        imageWidth = frame.shape[1]
        imageHeight = frame.shape[0]

        personLocations = []

        FaceLocationsBodyLocations = []
        FaceLocationsBodyIndexes = []

        FaceLocationsShoulderLocations = []
        FaceLocationsShoulderIndexes = []

        FaceLocationsPupilLocations = []
        FaceLocationsPupilIndexes = []

        BodyMidpointPixelLocations = []
        ShoulderMidpointPixelLocations = []
        PupilMidpointPixelLocations = []

        rect_index = 0
        for merged_rect_location in merged_rect_locations:

            rect_index = merged_rect_locations.index(merged_rect_location)
            YOLO_person_count = len(rectangle_groups[rect_index])

            height = merged_rect_location[3]-merged_rect_location[2]
            width = merged_rect_location[1]-merged_rect_location[0]

            box_image = np.zeros((height, width, 3), np.uint8)
            box_image[0:height, 0:width, :] = frame[merged_rect_location[2]:merged_rect_location[3],
                                                    merged_rect_location[0]:merged_rect_location[1], :]

            datum = op.Datum()
            datum.cvInputData = box_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            #transform the keypoints to their locations in the full image
            if datum.poseKeypoints is not None:
                keypoints = datum.poseKeypoints
                for i in range(keypoints.shape[0]):
                    keypoints[i][:, 0] += merged_rect_location[0]
                    keypoints[i][:, 1] += merged_rect_location[2]

                eyePairs = eyePairCount(keypoints)

                if eyePairs > YOLO_person_count:
                    keypoints = removeFaultyKeypoints(keypoints, eyePairs, YOLO_person_count)
                    eyePairs = eyePairCount(keypoints)

                faceLocationsPIndexes, BodyMidpointPixelLocation, FaceLocationsBodyLocation = body_distance(keypoints,
                                                                                                            rect_index,
                                                                                                            sensorWidth,
                                                                                                           sensorHeight,
                                                                                                            imageWidth,
                                                                                                            imageHeight,
                                                                                                            focalLength,
                                                                                                    realUpperBodyLength)
                FaceLocationsBodyLocations += FaceLocationsBodyLocation
                FaceLocationsBodyIndexes += faceLocationsPIndexes
                BodyMidpointPixelLocations += BodyMidpointPixelLocation

                faceLocationsPIndexes, ShoulderMidpointPixelLocation, \
                FaceLocationsShoulderLocation = shoulder_distance(keypoints, rect_index, sensorWidth, sensorHeight,
                                                                  imageWidth,
                                                                  imageHeight, focalLength, realShoulderWidth)
                FaceLocationsShoulderLocations += FaceLocationsShoulderLocation
                FaceLocationsShoulderIndexes += faceLocationsPIndexes
                ShoulderMidpointPixelLocations += ShoulderMidpointPixelLocation

                faceLocationsPIndexes, PupilMidpointPixelLocation, FaceLocationsPupilLocation = pupil_distance(keypoints
                                                                                                            ,rect_index,
                                                                                                            sensorWidth,
                                                                                                           sensorHeight,
                                                                                                            imageWidth,
                                                                                                            imageHeight,
                                                                                                            focalLength,
                                                                                                        realEyeDistance)
                FaceLocationsPupilLocations += FaceLocationsPupilLocation
                FaceLocationsPupilIndexes += faceLocationsPIndexes
                PupilMidpointPixelLocations += PupilMidpointPixelLocation

            output_box_image = datum.cvOutputData

            rect_index += 1

            frame[merged_rect_location[2]:merged_rect_location[3],
                  merged_rect_location[0]:merged_rect_location[1], :] = output_box_image

        minLocations = minimum_locations(FaceLocationsPupilLocations, FaceLocationsPupilIndexes,
                                           FaceLocationsShoulderLocations, FaceLocationsShoulderIndexes,
                                           FaceLocationsBodyLocations, FaceLocationsBodyIndexes)

        personLocations = minLocations

        personID = 0

        averageDistance = 0
        distCount = 0  # number of person pairs

        personLocations = sorted(personLocations, key=itemgetter(0))

        shoulderMidPoints = []
        pupilMidPoints = []
        bodyMidPoints = []
        headMidPoints = []

        for loc in personLocations:
            locationPIndex = ''
            try:
                locShoulderIndex = FaceLocationsShoulderLocations.index(loc)
                locationPIndex = FaceLocationsShoulderIndexes[locShoulderIndex]
            except:
                locShoulderIndex = -1

            try:
                locPupilIndex = FaceLocationsPupilLocations.index(loc)
                locationPIndex = FaceLocationsPupilIndexes[locPupilIndex]
            except:
                locPupilIndex = -1

            try:
                locTorsoIndex = FaceLocationsBodyLocations.index(loc)
                locationPIndex = FaceLocationsBodyIndexes[locTorsoIndex]
            except:
                locTorsoIndex = -1

            headMidPoint = [0, 0]

            try:
                shoulderMidPoint = ShoulderMidpointPixelLocations[FaceLocationsShoulderIndexes.index(locationPIndex)]
            except:
                shoulderMidPoint = [0, 0]

            try:
                pupilMidPoint = PupilMidpointPixelLocations[FaceLocationsPupilIndexes.index(locationPIndex)]
            except:
                pupilMidPoint = [0, 0]

            try:
                bodyMidPoint = BodyMidpointPixelLocations[FaceLocationsBodyIndexes.index(locationPIndex)]
            except:
                bodyMidPoint = [0, 0]

            shoulderMidPoints.append(shoulderMidPoint)
            pupilMidPoints.append(pupilMidPoint)
            bodyMidPoints.append(bodyMidPoint)
            headMidPoints.append(headMidPoint)

        auto_err = automatic_per_err(personLocations, shoulderMidPoints, pupilMidPoints, bodyMidPoints, headMidPoints,
                                     filename, imageWidth, imageHeight)

        detectedPersonCount = len(personLocations)

        detRate = detection_rate(detectedPersonCount, filename)
        falseDiscoveryRate = false_discovery_rate(detectedPersonCount, filename)

        violationCount = 0

        if (len(personLocations) > 1):
            for i in range(len(personLocations)):
                for k in range(i + 1, len(personLocations)):
                    dist = distanceBetweenTwoPoints(personLocations[i], personLocations[k])
                    distances.append(dist)
                    averageDistance += dist
                    if dist < safeDistance:
                        violationCount += 1
                    distCount += 1

        if distCount > 0:
            averageDistance /= distCount

        distancesString = ''
        for distance in distances:
            distancesString += str(distance) + ', '
        distancesString = distancesString[:len(distancesString) - 2]

        filenames.append(filename)
        detectedPersonCounts.append(detectedPersonCount)
        distancesStrings.append(distancesString)
        per_errors.append(auto_err)
        avg_distances.append(averageDistance)
        violation_counts.append(violationCount)
        detection_rates.append(detRate)
        false_discovery_rates.append(falseDiscoveryRate)

    if not os.path.exists('outputs_labeled_data'):
        os.makedirs('outputs_labeled_data')

    output_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = 'outputs_labeled_data/' + output_filename + '.csv'

    with open(output_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Detected Person Count", "Social Distances (mm)", "Average Pair-Wise Percentual Social Distance Estimation Error",
                             "Average Social Distance (mm)", "Violation Count", "Detection Rate", "False Discovery Rate"])
        for i in range(len(filenames)):
            writer.writerow([filenames[i], detectedPersonCounts[i], distancesStrings[i], per_errors[i], avg_distances[i], violation_counts[i],
                             detection_rates[i], false_discovery_rates[i]])
        writer.writerow(['ALL', averageOfList(detectedPersonCounts), "-", averageOfList(per_errors), averageOfList(avg_distances), averageOfList(violation_counts),
                         averageOfList(detection_rates), averageOfList(false_discovery_rates)])

if __name__ == "__main__":
   main()