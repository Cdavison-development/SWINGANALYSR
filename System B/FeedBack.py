import json
import math
import openai
#this system calculates the euclidean distance of the Top3 professional average and the amateur average, this data is entered into an OpenAI NLP prompt 
#which will return inference on the data
openai.api_key = "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"

# List of filenames
filenames = [
    "Address",
    "Mid-Backswing (arm parallel)",
    "Mid-Downswing (arm parallel)",
    "Mid-Follow-Through (shaft parallel)",
    "Top",
    "Toe-up",
    "Impact",
    "Finish",
]
#keypoint groups
keypoint_lists = {
    "LshoulderToElbow": [11, 13],
    "LelbowToWrist": [13,15],
    "LshoulderToElbow": [12, 14],
    "LelbowToWrist": [14,16],
    "LshouldertoHip": [11, 23],
    "RshouldertoHip": [12, 24],
    "LHipToKnee": [23, 25],
    "RHipToKnee": [24, 26],
    "LKneetoAnkle":[25,27],
    "RkneetoAnkle":[26,28]
}
#description of each keypoint
keypoint_descriptions = {
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
}

# Loop through filenames
for filename in filenames:
    
    with open(f'Pose_estimation/Keypoint_data/AmateurKeypointData/{filename}_keypointData.json', 'r') as f:
        data1 = json.load(f)

    with open(f'Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/{filename}/{filename}_averages.json', 'r') as f:
        data2 = json.load(f)

    #function for calculating euclidean distance
    def euclidean_distance(x1, y1, z1, x2, y2, z2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    #finds keypoint distances through euclidean distance 
    max_avg_distance = 0
    max_avg_distance_list_name = None
    max_avg_distance_original_values = []
    
    for list_name, keypoint_list in keypoint_lists.items():
        total_distance = 0
        list_original_values = []

        for key in keypoint_list:
            key = str(key)
            if key in data1 and key in data2:
                x1, y1, z1 = data1[key]
                x2, y2, z2 = data2[key]
                distance = euclidean_distance(x1, y1, z1, x2, y2, z2)
                total_distance += distance
                list_original_values.append(((x1, y1, z1), (x2, y2, z2)))

        avg_distance = total_distance / len(keypoint_list)

        if avg_distance > max_avg_distance:
            max_avg_distance = avg_distance
            max_avg_distance_list_name = list_name
            max_avg_distance_original_values = list_original_values

    

    #generates feedback by prompting text-davinci-003 with keypoint differences between amateur and professional average
    def generate_feedback(list_name, original_values):
        feedback = []

        for i, values in enumerate(original_values):
            amateur_coords, pro_coords = values
            keypoint = keypoint_lists[list_name][i]
            keypoint_description = keypoint_descriptions[keypoint]

            diff_x = amateur_coords[0] - pro_coords[0]
            diff_y = amateur_coords[1] - pro_coords[1]
            diff_z = amateur_coords[2] - pro_coords[2]
            #print(diff_x , diff_y , diff_z)
            prompt = (f"For the {keypoint_description} (keypoint {keypoint}), the amateur data coordinates are {amateur_coords}, and the professional data coordinates are {pro_coords}. "
                  f"The differences in x, y, and z are {diff_x}, {diff_y}, and {diff_z}, respectively. "
                  f"How can the user improve their golf swing based on this data?, do not return data that references specific x, y or z values, or the professional swing, instead make inferences based on these values")

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=400,
                n=1,
                stop=None,
                temperature=0.8,
                
                )

            feedback_item = response.choices[0].text.strip()
            feedback.append(feedback_item)
            
            #num_tokens = num_tokens_from_string(item, "p50k_base")
            #print(num_tokens)
        return feedback

    #feedback is returned
    feedback = generate_feedback(max_avg_distance_list_name, max_avg_distance_original_values)
    print("-------------------------------------------------------------------------------")
    print(f" Advice for : {filename}")
    for item in feedback:
        print("\n" + item)
    print("-------------------------------------------------------------------------------")
    #num_tokens = num_tokens_from_string(feedback, "p50k_base")
    #print(num_tokens)