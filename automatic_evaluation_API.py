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

def distanceBetweenTwoPoints(pos1 = [], pos2 = []):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
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
# also takes a safe distance (mm) as an input and output the number of violations where the social distances are
# smaller than the safe distance
# finally, the function returns the number of detected people,
# average of pair-wise percent distance estimation errors
# average social distance
# number of violations
# person detection rate
# false discovery rate
def automatic_evaluate(personLocations, ShoulderPixelPoints, PupilPixelPoints, BodyPixelPoints, HeadPixelPoints,
                      filename, imageWidth, imageHeight, safeDistance):

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

    averageDistance = 0
    distCount = 0
    violationCount = 0
    if (len(personLocations) > 1):
        for i in range(len(personLocations)):
            for k in range(i + 1, len(personLocations)):
                dist = distanceBetweenTwoPoints(personLocations[i], personLocations[k])
                averageDistance += dist
                if dist < safeDistance:
                    violationCount += 1
                distCount += 1

    if distCount > 0:
        averageDistance /= distCount

    return len(personLocations), per_err, averageDistance, violationCount, \
           detection_rate(len(personLocations), filename), false_discovery_rate(len(personLocations), filename)


# same as automatic_per_err but instead of 3D locations, the function takes the number of people
# and the estimated distances between them as inputs
# if there are n people in the picture, they would be indexed 0,1,2,3,...,n for this function
# socialDistances should be a list of all the pair-wise estimated distances
# the order of the distances should be in the following order of human pairs:
# 0-1, 0-2, ..., 0-n, 1-2, 1-3, ..., 1-n, 2-3, 2-4, ..., 2-n, ..., (n-1)-n
# the 4 body pixel point arrays should contain the respective body pixel locations of the people in the following human
# index order: 0,1,2,3,...,n
def automatic_evaluate_2(numberOfPeople, socialDistances, ShoulderPixelPoints, PupilPixelPoints, BodyPixelPoints,
                        HeadPixelPoints, filename, imageWidth, imageHeight, safeDistance):

    if numberOfPeople < 2:
        return 0, 0, 0, 0, 0, 0

    if math.comb(numberOfPeople, 2) != len(socialDistances):
        return 0, 0, 0, 0, 0, 0

    #there must be at least 1 body pixel location provided for a person
    #if all three body pixel locations are missing for a given person location, matching cannot be done
    for locIndex in range(numberOfPeople):
        if ShoulderPixelPoints[locIndex] == [0, 0] and PupilPixelPoints[locIndex] == [0, 0] and BodyPixelPoints[locIndex] == [0, 0] and HeadPixelPoints[locIndex] == [0, 0]:
            return 0, 0, 0, 0, 0, 0

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
            return 0, 0, 0, 0, 0, 0

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
        return 0, 0, 0, 0, 0, 0

    per_err = pairwise_distance_evaluation_2(pair_wise_dist_est_dict, photoshoot_id)

    violationCount = 0

    for socialDistance in socialDistances:
        if socialDistance < safeDistance:
            violationCount += 1

    averageDistance = sum(socialDistances) / len(socialDistances)

    return numberOfPeople, per_err, averageDistance, violationCount, \
           detection_rate(numberOfPeople, filename), false_discovery_rate(numberOfPeople, filename)

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
        # the following codes are only meant to show example cases of how these automatic evaluation functions
        # can be used

        # for the functions to work, all the necessary information
        # about the image to be evaluated must be present in the following three csv files:
        # body_pixel_locations.csv
        # camera_locations_photoshoot_idnetifiers.csv
        # ground_truth_locations.csv

        # the csv files should be under a directory called "labels" that is in the same sub directory as the code
        # that would use these functions

        # the values of the variables are only meant as placeholders

        # automatic_evaluation() is meant to be used by methods that can
        # estimate 3D locations of all the people in the images
        # and also provide at least one pixel location of the following four body parts:
        # center of eyes, shoulders, torso and head
        # the order of the 3D location estimations and the body pixel points should correspond to the same people
        # i.e., n'th element of any list should be information about the same person


        # example output of a method that produces 3D locations as the following list
        personLocations = [[-2400, 0, 6000], [-1800, 0, 6000], [-1200, 0, 4800]]

        # pixel body locations of the detected people
        # in this case, only the shoulder points are provided
        # missing points should be filled as [0, 0]
        # it is enough that at least 1 body pixel point is provided for a person
        shoulderPoints = [[1060, 840], [2280, 660], [3050, 830]]
        pupilPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]
        bodyPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]
        headPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # filename should only be the filename and not the full path
        filename = "test.jpg"
        imageWidth = 4000
        imageHeight = 2000

        safeDistance = 2000  # mm


        # the function outputs the number of detected people, average percentual social distance estimation error,
        # average social distance between the detected people, number of violations, person detection rate
        # and the false discovery rate
        numOfPeople, perErr, avgDist, vioCount, detRate, falseDiscRate = \
            automatic_evaluate(personLocations, shoulderPoints, pupilPoints, bodyPoints, headPoints, filename,
                               imageWidth, imageHeight, safeDistance)

        ###############################################################################################################

        # automatic_evaluation_2() is meant to be used by methods that can directly
        # estimate distances between the people in the images
        # and also provide at least one pixel location of the following four body parts:
        # center of eyes, shoulders, torso and head

        # if there are n detected people in the picture, the indexes of the people can be thought as:
        # 0, 1, 2, 3, 4, ..., n
        # the method should should provide body pixel points in the corresponding ascending order of these indexes
        # given this information,
        # the order of the list that contains pair-wise estimated distances must be in this pair order:
        # 0-1, 0-2, 0-3, ... , 0-n, 1-2, 1-3, 1-4, ... , 1-n, ..... , (n-1)-n

        # an example case can be seen below

        filename = "test.jpg"
        imageWidth = 4000
        imageHeight = 2000

        numPeople = 3
        # these social distance belongs to the following pairs in order : 0-1, 0-2, 1-2
        socialDistances = [900, 1690, 2700]

        # these body pixel points belongs to the people with the following indexes: 0, 1, 2
        shoulderPoints = [[1060, 840], [2280, 660], [3050, 830]]
        pupilPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]
        bodyPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]
        headPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]


        safeDistance = 2000

        # the function outputs the number of detected people, average percentual social distance estimation error,
        # average social distance between the detected people, number of violations, person detection rate
        # and the false discovery rate
        numPeople, perErr, avgDist, vioCount, detRate, falseDiscRate = automatic_evaluate_2(numPeople, socialDistances,
                                                                                            shoulderPoints, pupilPoints,
                                                                                            bodyPoints, headPoints,
                                                                                            filename, imageWidth,
                                                                                            imageHeight,
                                                                                            safeDistance)

if __name__ == "__main__":
   main()