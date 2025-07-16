import os
import cv2
import json
import numpy as np
import configparser
import geopandas as gpd
import pandas as pd
import pickle
# from PIL import Image
# from osgeo import ogr
from osgeo import ogr
from PIL import Image
import yaml

## 以cqs_为前缀的读写函数只被CrowdQualitySample类调用
# 1. 明确作用域：新命名方式明确标识了这些函数与 CrowdQualitySample 的关联，避免与其他 I/O 函数冲突。
# 2. 易于维护：在 io/file_io.py 中可以轻松识别与 CrowdQualitySample 相关的函数。
# 3. 统一性：用户看到 cqs_ 前缀时即可明确这些函数适用于 CrowdQualitySample。
def cqs_load_geojson(file_path):
    """加载 CrowdQualitySample 的 GeoJSON 文件"""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def cqs_load_csv(file_path):
    """加载 CrowdQualitySample 的 CSV 文件"""
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def cqs_list_image_paths(dir_path):
    """列出 CrowdQualitySample 目录中的所有图像文件路径"""
    if dir_path and os.path.isdir(dir_path):
        return [
            os.path.join(dir_path, file_name)
            for file_name in os.listdir(dir_path)
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp','.pkl'))
        ]
    return []

def cqs_load_images(image_paths):
    """加载 CrowdQualitySample 指定路径的所有图像"""
    images = {}
    for file_path in image_paths:
        file_name = os.path.basename(file_path)
        images[file_name] = Image.open(file_path).convert('RGB')
    return images


def read_semantic_slice_to_ground_truth_dict(_pkl_file_name:str):
    with open(_pkl_file_name, 'rb') as f:
        _slice_to_gt_tile = pickle.load(f)
    return _slice_to_gt_tile
def _m_read_data_to_ogr_datasource_batch(data_name):
    if os.path.isdir(data_name):
        print("数据读取 - 该要素输入形式为文件夹\n")
    feature_data_source_dict = {}
    _slice_list = os.listdir(data_name)

    for _slice in _slice_list:
        _key = _slice.split('.')[0]
        _slice_full_path = os.path.join(data_name,_slice)
        feature_data_source = _m_read_data_to_ogr_datasource(_slice_full_path)
        feature_data_source_dict[_key] = feature_data_source
        print("车端数据slice数量：",len(feature_data_source_dict),'/',len(_slice_list),end = '\r')
    return feature_data_source_dict

# 原始caq_utd工程中使用数据的方式
def _m_read_data_to_ogr_datasource(data_name):
    #首先判断dataname是否包含路径
    folder_name,file_name= os.path.split(data_name)
    file_name,file_ext = os.path.splitext(file_name)

    with open(data_name,'r',encoding='utf-8') as fp:
        _content = fp.read()
        # _feature_data_source = read_geojson(_content)
        _feature_data_source = ogr.Open(_content)
        if _feature_data_source is None:
            print(f"无法打开文件：{_content}")
            return None
        else:
            total_count = 0
            _lyr_count = _feature_data_source.GetLayerCount()
            for i in range(_lyr_count):
                _feat_num = _feature_data_source[i].GetFeatureCount()
                total_count += _feat_num
            if total_count == 0:
                return None
            return _feature_data_source

def read_loc_data(path):
    df = pd.read_csv(path)    
    return df
    
#此类别中需要编写关于读取

def read_mask_data(path,mask_type="pkl"): 
    if mask_type == "pkl":
        with open(path, 'rb') as f:
        # 使用pickle.load加载数据
            loaded_data = pickle.load(f)
        return loaded_data
    elif mask_type=="jpg":
        image_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image_array

def read_vec_data(path):
    with open(path,"r") as fp:
        _content = fp.read()
        print(_content)
        return _content

def read_vehicle_type(config_file_path):
    if config_file_path.endswith('yaml'):
        with open(config_file_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
        vt = config.get('wrap_setting').get('vehicle_type')
        return vt
    else:
        return None

def read_config_file(config_file_path):
    if config_file_path.endswith('ini'):
        config = configparser.ConfigParser()
        config.read(config_file_path,encoding = 'UTF8')

        data_root = config.get('Paths','data_root')

        feature_file_path_rela = config.get('Paths','feature_file_path')
        location_file_path_rela = config.get('Paths','location_file_path')
        vision_file_path_rela = config.get('Paths','vision_file_path')

        feature_file_path = os.path.join(data_root,feature_file_path_rela)
        location_file_path = os.path.join(data_root,location_file_path_rela)
        vision_file_path = os.path.join(data_root,vision_file_path_rela)
    elif config_file_path.endswith('yaml'):
        with open(config_file_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
        data_root = config['wrap_setting']['data_root']

        feature_file_path_rela = config['wrap_setting']['feature_file_path']
        location_file_path_rela = config['wrap_setting']['location_file_path']
        vision_file_path_rela = config['wrap_setting']['vision_file_path']
        
        feature_file_path = os.path.join(data_root,feature_file_path_rela)
        location_file_path = os.path.join(data_root,location_file_path_rela)
        vision_file_path = os.path.join(data_root,vision_file_path_rela)
    return [feature_file_path,location_file_path,vision_file_path]

def read_config_file_valid(config_file_path,if_yaml=True):
    if if_yaml == True:
        with open(config_file_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)

        gt_tile = config['gt_label_setting'].get('gt_tile', None)
        gt_file_path = config['gt_label_setting'].get('gt_file_path', None)
        return [gt_tile,gt_file_path]
    else:    
        config = configparser.ConfigParser()
        config.read(config_file_path,encoding = 'UTF8')

        gt_feature_path = config.get('Paths','GROUND_TRUTH_FEATURE_TILED_PATH')
        data_root = config.get('Paths','data_root')
        ground_truth_tile_folder = config.get('Paths','ground_truth_tile_folder')
        
        ground_truth_tile_folder_full_path = os.path.join(data_root,ground_truth_tile_folder)

        return [ground_truth_tile_folder_full_path,gt_feature_path]
    
def read_config_file_validCD701(config_file_path,if_yaml=True):
    if if_yaml == True:
        with open(config_file_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)

        gt_tile = config['gt_label_setting_CD701'].get('gt_tile', None)
        gt_file_path = config['gt_label_setting_CD701'].get('gt_file_path', None)
        return [gt_tile,gt_file_path]
    else:    
        config = configparser.ConfigParser()
        config.read(config_file_path,encoding = 'UTF8')

        gt_feature_path = config.get('Paths','GROUND_TRUTH_FEATURE_TILED_PATH')
        data_root = config.get('Paths','data_root')
        ground_truth_tile_folder = config.get('Paths','ground_truth_tile_folder')
        
        ground_truth_tile_folder_full_path = os.path.join(data_root,ground_truth_tile_folder)

        return [ground_truth_tile_folder_full_path,gt_feature_path]
    
def read_sample_geojson_file(geojson_path):
    gdf_vec = gpd.read_file(geojson_path)
    return gdf_vec

def read_sample_location_file(location_paths):
    location_paths = [location_paths] if isinstance(location_paths, str) else location_paths
    df_loc_set = {}
    for _index in range(len(location_paths)):
        df_loc = pd.read_csv(location_paths[_index])
        df_loc_set[_index] = df_loc
    
    merged_df = pd.DataFrame()

    # 迭代地合并 DataFrame
    for df in df_loc_set.values():
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

def raed_feather(feather_path):
    #此函数用用户读取相机的内部参数与外部参数
    with open(feather_path,"rb") as fp:
        _content = fp.read()
    return _content