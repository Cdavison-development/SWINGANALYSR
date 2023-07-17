import cv2 as cv2
import mediapipe as mp
import os 
import numpy as np
import json
#This system writes the mean amateur keypoint data of the amateur to a JSON file 
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#keypoint list
keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

#folder directories
folders = ['../golfdb/ImageData/Am_Swing_events/Address', 
           '../golfdb/ImageData/Am_Swing_events/Mid-Backswing (arm parallel)', 
           '../golfdb/ImageData/Am_Swing_events/Mid-Downswing (arm parallel)', 
           '../golfdb/ImageData/Am_Swing_events/Mid-Follow-Through (shaft parallel)', 
           '../golfdb/ImageData/Am_Swing_events/Impact', 
           '../golfdb/ImageData/Am_Swing_events/Top', 
           '../golfdb/ImageData/Am_Swing_events/Finish',
           '../golfdb/ImageData/Am_Swing_events/Toe-up']

#defining axes variables for each folder
for folder in folders:
    x_data_values = [[] for i in range(len(keypoints))]
    y_data_values = [[] for i in range(len(keypoints))]
    z_data_values = [[] for i in range(len(keypoints))]
    
    #openCV reads in image file and Mediapipe pose() function applies pose estimation techniques
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename) 
        img = cv2.imread(f)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #wstores data for each landmark
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id+1 in keypoints:
                    x_data_values[keypoints.index(id+1)].append(lm.x)
                    y_data_values[keypoints.index(id+1)].append(lm.y)
                    z_data_values[keypoints.index(id+1)].append(lm.z)
    #writes data to json file 
    with open(f'Keypoint_data/AmateurKeypointData/{os.path.basename(folder)}_keypointData.json', 'w') as file:
        keypoint_data = {}
        for id, keypoint in enumerate(keypoints):
            avg_x = np.mean(x_data_values[id])
            avg_y = np.mean(y_data_values[id])
            avg_z = np.mean(z_data_values[id])
            keypoint_data[keypoint] = [avg_x, avg_y, abs(avg_z)]
        json.dump(keypoint_data, file, indent=4)

        # print the average values for each keypoint
        for keypoint, data in keypoint_data.items():
            print(f"{os.path.basename(folder)} - Keypoint {keypoint}: x: {data[0]}, y: {data[1]}, z: {data[2]}")