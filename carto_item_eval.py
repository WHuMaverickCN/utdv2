import yaml
import os
import json
from src.hd_data import MapLearningData

CARTOGRAPHY_PREFIX = {
    "CD701-000052-20240412": "/data/gyx/cqc_p3_raw/map_learin_datasets/CD701-000052-20240412对应云端数据",
    "CD701-000013-20240501": "/data/gyx/cqc_p3_raw/map_learin_datasets/CD701-000013-2024-0501-0507-0508对应云端数据",
    "CD701-000013-20240507": "/data/gyx/cqc_p3_raw/map_learin_datasets/CD701-000013-2024-0501-0507-0508对应云端数据",
    "CD701-000013-20240508": "/data/gyx/cqc_p3_raw/map_learin_datasets/CD701-000013-2024-0501-0507-0508对应云端数据",
}

CARTOGRAPHY_TO_RECONS_PREFIX = {
    "/data/gyx/cqc_p3_raw/map_learin_datasets/CD701-000052-20240412对应云端数据": \
        "/data/gyx/cqc_p2_cd701_raw/datasets_A4_0326"
}

def get_cartography_dataset_path(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 获取utdv2_settings下的cartography_dataset路径
    carto_path = config['utdv2_settings']['target_maplearn_data_dir']
    geojson_paths = []
    for folder in os.listdir(carto_path):
        folder_path = os.path.join(carto_path, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            geojson_path = os.path.join(folder_path, "HADLaneDivider.geojson")
            if os.path.isfile(geojson_path):
                geojson_paths.append(geojson_path)

    candidate_paths = config['utdv2_settings']['candidate_recons_data_path']
    return geojson_paths,candidate_paths


def view_properties(target_path: str):
    with open(target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    features = data.get('features', [])
    for idx, feature in enumerate(features):
        properties = feature.get('properties', {})
        print(f"Feature {idx}:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
        print("-" * 40)
if __name__ == "__main__":
    dataset_path,candidate_paths = get_cartography_dataset_path()
    print(len(dataset_path), "cartography dataset files found.")
    # print(dataset_path[0])
    target_maplearn_data_dir = os.path.dirname(dataset_path[0])
    for stri in dataset_path:
        if '556162990' in stri:
            target_maplearn_data_dir = os.path.dirname(stri)


    maplearn_data = MapLearningData(data_dir = target_maplearn_data_dir, \
                                ground_truth_dir = 'gt_dir')
    maplearn_data.init_transform()
    maplearn_data.search_overlap_lane_pointset_in_candidate(candidate_paths)

    # view_properties(dataset_path[0])  # 查看第一个geojson文件的属性
    # print(f"Cartography dataset path: {dataset_path}")