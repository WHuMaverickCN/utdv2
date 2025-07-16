import pandas as pd
import shutil
import geojson
import os
import json

def write_recons_bias_info(target_file_path,
                    bias_info,
                    bias_file_name = 'veh2recons_bias.json'):
    target_dir = os.path.dirname(target_file_path)
    output_json_path = os.path.join(target_dir, bias_file_name)
    with open(output_json_path, 'w') as json_file:
        json.dump(bias_info, json_file, indent=4)
    

def write_bias_info(target_file_path,
                    bias_info,
                    mode):
    """
    Write a dataframe to csv file.
    :param df: Dataframe
    :type df: pandas.DataFrame
    :param path: Path of the output file
    :type path: str
    """
    target_dir = os.path.dirname(target_file_path)
    bias_file_name = mode + '_bias.json'
    output_json_path = os.path.join(target_dir, bias_file_name)
    with open(output_json_path, 'w') as json_file:
        json.dump(bias_info, json_file, indent=4)

    # print(f"Dictionary has been written to {output_json_path}")

def write_reconstructed_result(default_output_path,
                        output_file_name,
                        feature_collection,
                        recons_log,
                        IF_OPTIMIZE=True):
    """
    Write a dataframe to csv file.
    :param df: Dataframe
    :type df: pandas.DataFrame
    :param path: Path of the output file
    :type path: str
    """
    if IF_OPTIMIZE:
        with open(os.path.join(default_output_path, output_file_name+".geojson"), 'w') as f:
            geojson.dump(feature_collection, f)
    
        with open(os.path.join(default_output_path, output_file_name+"_recons_log.json"), 'w') as f:
            geojson.dump(recons_log, f)
        # with open(os.path.join(default_output_path, output_file_name+"_opt.geojson"), 'w') as f:
        #     geojson.dump(feature_collection, f)
    
        # with open(os.path.join(default_output_path, output_file_name+"_opt_recons_log.json"), 'w') as f:
        #     geojson.dump(recons_log, f)
    else:
        with open(os.path.join(default_output_path, output_file_name+"_raw.geojson"), 'w') as f:
            geojson.dump(feature_collection, f)
    
        with open(os.path.join(default_output_path, output_file_name+"_raw_recons_log.json"), 'w') as f:
            geojson.dump(recons_log, f)

def write_to_foler(
                   dat_name,#dat包文件名
                   loc_data_2_vis_data,# 定位数据与视觉数据的对应关系
                   target_folder='output'):# 目标路径
    # 检查目标文件夹是否存在，如果不存在则创建它
    target_sample_folder = os.path.join(target_folder,dat_name) 
    if not os.path.exists(target_sample_folder):
        os.makedirs(target_sample_folder)
    
    loc_data_2_vis_data.to_csv(target_sample_folder+'/loc2vis.csv')
    return target_sample_folder+'/loc2vis.csv'


def BAN_write_to_foler(full_vec_path,#  矢量全路径
                   full_traj_path,# 轨迹全路径
                   fetched_loc_data_frame,#筛选到的定位数据dataframe
                   loc_data_2_vis_data,
                   vis_data_dict,
                   target_folder='output',
                   index='temp'):# 目标路径
    # 检查目标文件夹是否存在，如果不存在则创建它
    target_sample_folder = os.path.join(target_folder,str(index)) 
    if not os.path.exists(target_sample_folder):
        os.makedirs(target_sample_folder)

    shutil.copy(full_vec_path,target_sample_folder)
    shutil.copy(full_traj_path,target_sample_folder)
    fetched_loc_data_frame.to_csv(target_sample_folder+'/loc_data.csv')
    
    loc_data_2_vis_data.to_csv(target_sample_folder+'/loc2vis.csv')

    for _key in vis_data_dict.keys():
        if _key=="cam_1":
            continue
        _target_root = os.path.join(target_sample_folder, _key)
        if not os.path.exists(_target_root):
            os.makedirs(_target_root)

        for _path in vis_data_dict[_key]:
            _target_path = os.path.join(target_sample_folder,_key)
            shutil.copy(_path,_target_path)

    