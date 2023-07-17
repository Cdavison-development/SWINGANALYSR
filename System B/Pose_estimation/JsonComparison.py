import cv2 as cv2
import mediapipe as mp
import os 
import numpy as np
import json
#This system Finds the top 3 golfers most similar to amateur average and writes the keypoint data for each moment of swing to a file

mpPose = mp.solutions.pose
pose = mpPose.Pose()

keypoints = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]

# load amateur keypoint data
amateur_keypoint_files = {
    'Address': 'Keypoint_data/AmateurKeypointData/Address_keypointData.json',
    'Mid-Backswing (arm parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Backswing (arm parallel)_keypointData.json',
    'Mid-Downswing (arm parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Downswing (arm parallel)_keypointData.json',
    'Mid-Follow-Through (shaft parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Follow-Through (shaft parallel)_keypointData.json',
    'Top': 'Keypoint_data/AmateurKeypointData/Top_keypointData.json',
    'Toe-up': 'Keypoint_data/AmateurKeypointData/Toe-up_keypointData.json',
    'Impact': 'Keypoint_data/AmateurKeypointData/Impact_keypointData.json',
    'Finish': 'Keypoint_data/AmateurKeypointData/Finish_keypointData.json'
}

average_distances = []
amateur_keypoint_data = {}
for moment, filepath in amateur_keypoint_files.items():
    with open(filepath, 'r') as file:
        amateur_keypoint_data[moment] = json.load(file)

best_player = ''
best_distance = float('inf')

#loop through each professional
for i in range(1, 226):
    print("Processing player", i)
    #load keypoint data for each professional
    professional_keypoint_data = {}
    for moment in amateur_keypoint_files.keys():
        filename = f'{moment}{i}.json'
        filepath = os.path.join(f'Keypoint_data/ProfessionalKeypointData/{moment}', filename)
        
        with open(filepath, 'r') as file:
            professional_keypoint_data[moment] = json.load(file)

    #calculate euclidean distance between amateur and professionals per moment of swing
    total_distance = 0
    for moment in amateur_keypoint_files.keys():
        amateur_data = amateur_keypoint_data[moment]
        professional_data = professional_keypoint_data[moment]

        distance = 0
        for keypoint, amateur_keypoint_values in amateur_data.items():
            professional_keypoint_values = professional_data[keypoint]
            distance += np.linalg.norm(np.array(amateur_keypoint_values) - np.array(professional_keypoint_values))

        total_distance += distance / len(amateur_data)

    average_distance = total_distance / len(amateur_keypoint_files.keys())
    
    print("Average distance for player {}: {}".format(i, average_distance))
    average_distances.append((i, average_distance))

#sort the average distances to decending order
sorted_distances = sorted(average_distances, key=lambda x: x[1])
Top3List = []
#print 3 closest matches
print("Top 3 closest matches:")
for i in range(3):
    player, distance = sorted_distances[i]
    print("Player {}, Average distance: {}".format(player, distance))
    Top3List.append(player)
 

for i in range(1, 226):
    if i in Top3List:
        #load keypoint for each moment of swing
        for moment in amateur_keypoint_files.keys():
            filename = f'{moment}{i}.json'
            filepath = os.path.join(f'Keypoint_data/ProfessionalKeypointData/{moment}', filename)
            Top3filepath = os.path.join(f'Keypoint_data/Top3ProfessionalKeypointData/{moment}', filename)
            
            #read json file 
            with open(filepath, 'r') as file:
                professional_keypoint_data = json.load(file)
            
            #write JSON data to directory
            with open(Top3filepath, 'w') as file:
                json.dump(professional_keypoint_data, file)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
