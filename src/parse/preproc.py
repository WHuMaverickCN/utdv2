import os
import yaml
import json
_path = 'packages/map_recv_current_interface'

# Remove all .csv files in the _path directory
csv_files = [os.path.join(_path, f) for f in os.listdir(_path) if f.endswith('.csv')]
for csv_file in csv_files:
    os.remove(csv_file)

def get_dataset_path():
    config_path = './config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('dataset_path')

_path = get_dataset_path()
dats_folder = 'dats'
dats_path = os.path.join(_path, dats_folder)

# Get all .dat files in the dats_path directory
dat_files = [os.path.join(dats_path, f) for f in os.listdir(dats_path) if f.endswith('.dat')]

# Write the absolute paths of .dat files to a JSON file
output_file =  './dat_files.json'
with open(output_file, 'w') as json_file:
    json.dump(dat_files, json_file, indent=4)
print("[解析状态] 已将.dat文件的绝对路径写入到dat_files.json文件中")