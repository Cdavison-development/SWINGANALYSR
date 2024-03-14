import os
import shutil

def clean_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files and links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directories
            print(f"Removed: {item_path}")
        except Exception as e:
            print(f"Failed to remove {item_path}. Reason: {e}")

def clean_folders(folder_list):
    for folder_path in folder_list:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Cleaning folder: {folder_path}")
            clean_folder(folder_path)
        else:
            print(f"Skipping non-existent or non-directory path: {folder_path}")

# List of folder paths you want to clean
folders_to_clean = ['golfdb/ImageData/Am_Swing_events/Address', 
           'golfdb/ImageData/Am_Swing_events/Mid-Backswing (arm parallel)', 
           'golfdb/ImageData/Am_Swing_events/Mid-Downswing (arm parallel)', 
           'golfdb/ImageData/Am_Swing_events/Mid-Follow-Through (shaft parallel)', 
           'golfdb/ImageData/Am_Swing_events/Impact', 
           'golfdb/ImageData/Am_Swing_events/Top', 
           'golfdb/ImageData/Am_Swing_events/Finish',
           'golfdb/ImageData/Am_Swing_events/Toe-up',
           'golfdb/VideoData/Amateur-vids',
           'Pose_estimation/Keypoint_data/AmateurKeypointData',
           'Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData']

clean_folders(folders_to_clean)

