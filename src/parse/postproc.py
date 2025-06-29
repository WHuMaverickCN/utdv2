import os
import shutil
import yaml
import datetime
import pandas as pd


img_out = './img_out'
for root, dirs, files in os.walk(img_out):
    for dir_name in dirs:
        if dir_name == "sda-encode_srv-EncodeSrv.EncodeH265-CAM8":
            dir_path = os.path.join(root, dir_name)
            parent_dir = os.path.dirname(dir_path)
            
            # Move all files from the subdirectory to the parent directory
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    print(file_path, parent_dir)
                    shutil.move(file_path, parent_dir)
            
            # Remove the now-empty subdirectory
            os.rmdir(dir_path)

# Read dataset_path from the config.yaml file
with open("./config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
    dataset_root = config.get("dataset_path")

location_target_path = os.path.join(dataset_root, "location")
# Define column names
column_names = [
    'sec_of_week', 'gps_week_number', 'utc', 'position_type', 'numsv',
    'ins_status', 'temperature', 'latitude', 'longitude', 'altitude',
    'north_velocity', 'east_velocity', 'ground_velocity', 'roll', 'pitch',
    'heading', 'x_angular_velocity', 'y_angular_velocity',
    'z_angular_velocity', 'x_acc', 'y_acc', 'z_acc', 'latitude_std',
    'longitude_std', 'altitude_std', 'north_velocity_std',
    'east_velocity_std', 'ground_velocity_std', 'roll_std', 'pitch_std',
    'heading_std', 'atb_0', 'atb_1', 'atb_2', 'q_x', 'q_y', 'q_z', 'q_w',
    'fl_wheel_vel', 'fr_wheel_vel', 'rl_wheel_vel', 'rr_wheel_vel',
    'l_wheel_factor', 'r_wheel_factor', 'time_stamp', 'seq', 'time_valid',
    'timestamp_sab_vehicle', 'x_angular_velocity_bias',
    'y_angular_velocity_bias', 'z_angular_velocity_bias', 'x_acc_bias',
    'y_acc_bias', 'z_acc_bias', 'gps_velocity', 'wheel_speed', 'dv'
]

# Process CSV files in the location_target_path directory
for file_name in os.listdir(location_target_path):
    file_path = os.path.join(location_target_path, file_name)
    if file_name.endswith('.csv') and os.path.isfile(file_path):
        df_firstrow = pd.read_csv(file_path, nrows=1)

        if df_firstrow.columns.tolist() == column_names:
            location_data = pd.read_csv(file_path)
        else:
            # Read the original CSV file without headers
            df = pd.read_csv(file_path, header=None)
            # Assign the defined column names to the DataFrame
            df.columns = column_names
            df.to_csv(file_path, index=False)
            location_data = pd.read_csv(file_path)

# Construct the target directory
dataset_path = os.path.join(dataset_root, "vision")

# Ensure the target directory exists
os.makedirs(dataset_path, exist_ok=True)

# Move all files from img_out to the target directory
for dir_name in os.listdir("./img_out"):
    dir_path = os.path.join("./img_out", dir_name)
    if os.path.isdir(dir_path):
        shutil.move(dir_path, dataset_path)

print("[解析状态] 已将所有文件移动到目标目录")

# Extract the date from the dataset_root folder name
root_folder_name = os.path.basename(dataset_root)
date_part = root_folder_name.split("_")[2]  # Assuming the date is the first part of the folder name

# input()
# Construct new folder names
route_name = "routex"  # Assuming a fixed route name
location_folder_name = f"{date_part[:2]}-{date_part[2:]}_{route_name}_Location"
vision_folder_name = f"{date_part[:2]}-{date_part[2:]}_{route_name}"

# Create new directories
location_target_path = os.path.join(dataset_root, "location", location_folder_name)
vision_target_path = os.path.join(dataset_root, "vision", vision_folder_name)

os.makedirs(location_target_path, exist_ok=True)
os.makedirs(vision_target_path, exist_ok=True)

# Move contents of dataset_root/location to the new location folder
location_path = os.path.join(dataset_root, "location")
for item in os.listdir(location_path):
    item_path = os.path.join(location_path, item)
    shutil.move(item_path, location_target_path)
    print(item_path, location_target_path)

# Move contents of dataset_root/vision to the new vision folder
vision_path = os.path.join(dataset_root, "vision")
for item in os.listdir(vision_path):
    item_path = os.path.join(vision_path, item)
    shutil.move(item_path, vision_target_path)
    print(item_path, vision_target_path)

print("[解析状态] 已根据日期和路线名称重新组织文件夹")

