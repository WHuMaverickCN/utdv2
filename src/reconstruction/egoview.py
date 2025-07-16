import os
import cv2
import numpy as np
import math
import yaml
from pathlib import Path
from math import pi
import sympy as sp
from scipy.spatial.distance import euclidean
import glob
import re
from dataclasses import dataclass
from .landmark_utils import *    
import re

from .transformation_utils import *
from .seg_utils import run_segmentation
from .wrap_utils import DataSamplePathWrapper


from src.io import m_input as input
from src.io.m_output import write_reconstructed_result

from src.common_utils import CoordProcessor

np.set_printoptions(precision=6, suppress=True)

DEFAULT_MASK_FILE_PATH = "output/4/array_mask"
DEFAULT_PROCESSED_DAT_PATH = "output/4"
location_data_path = "output/4/loc2vis.csv"

# OPTIMIZE = True

class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
'''
class EgoviewReconstructionCD701:
    def __init__(self):
        self.OPTIMIZE = OPTIMIZE
        self.config_path = config_path
        # Load config from yaml file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract paths from config
        try:
            self.array_mask_path = self.config['reconstruction']['target_seg_files_path']
            # self.processed_dat_path = self.config['paths']['processed_dat_path'] 
            # self.loc_data_path = self.config['paths']['location_data_path']
        except KeyError as e:
            print(f"Missing required path in config: {e}")
            raise

        CD701_cam0_info = {
            "Calibration_Time":"23-09-25-11:20:03",
            "camera_x":0.0,
            "camera_y":0.0,
            "camera_z":1.4800000190734863,
            "center_u":1903.890015,
            "center_v":1065.380005,
            "distort":[0.64,-0.0069,0.000654,0.000118,-0.0057,1.0039,0.1319,-0.02019999921321869],
            "focal_u":1907.819946,
            "focal_v":1907.670044,
            "fov":120.0,
            "homegrahpy":
            [1.034273624420166,0.013729095458984375,-11.037109375,\
             -0.008372306823730469,1.033010482788086,134.24853515625,\
                9.313225746154785e-10,-3.725290298461914e-09,0.9999980926513672],
            "image_height":2160,
            "image_width":3840,
            "intern":1,
            "loc":0,
            "mask":3,
            "pitch":-0.0015706592239439487,
            "roll":0.0038397256284952164,
            "type":0,
            "valid_height":[1576,1576],
            "vcs":{"rotation":[0.0,0.0,0.0],
                   "translation":[3.106,1.143,0.0]},
            "vendor":"ar0820",
            "version":1,
            "yaw":0.008028507232666016
        }

        # 畸变系数基本没有变化，仅有小数点后几位的差异
        self.dist_coeffs_cam0 = np.array([0.64,-0.0069,0.000654,0.000118,-0.0057,1.0039,0.1319,-0.02019999921321869])
        
        # 内参未发生变化
        self.K_cam0 = np.array([[1907.819946, 0, 1903.890015],
                                [0, 1907.670044, 1065.380005],
                                [0, 0, 1]])

        self.cameraInfo = Info({
            "focalLengthX": int(1907.819946),   # focal length x #沿用C385
            "focalLengthY": int(1907.670044),   # focal length y 沿用C385
            "opticalCenterX": int(1903.890015), # optical center x 沿用C385
            "opticalCenterY": int(1065.380005), # optical center y 沿用C385
            "cameraHeight": 1576,    # camera height in `mm`  改动，使用"valid_height"字段值但存疑
            "pitch": -0.0015706592239439487*(180/pi),    # rotation degree around x  改动，使用新pitch值
            "yaw": 0.008028507232666016*(180/pi),   # rotation degree around y 改动，使用新yaw值
            "roll": 0.0038397256284952164*(180/pi),  # rotation degree around z 改动，使用新roll值
            "k1":0.64, # 畸变系数，基本沿用C385
            "k2":-0.0069, # 畸变系数，基本沿用C385
            "p1":0.000654, # 畸变系数，基本沿用C385
            "p2":0.000118, # 畸变系数，基本沿用C385
            "k3":-0.0057, # 畸变系数，基本沿用C385
            "k4":1.0039, # 畸变系数，基本沿用C385
            "k5":0.1319, # 畸变系数，基本沿用C385 
            "k6":-0.02019999921321869, # 畸变系数，基本沿用C385
            "tx":1.77,
            "ty":0.07,
            "tz":1.34,
            "rx":-0.0015706592239439487,
            "ry":0.008028507232666016+pi/2,
            "rz":0.0038397256284952164-pi/2
        })
'''
# Define regex pattern for CD701 files
PATTERN_701 = r"CD701_\d{6}_{1,2}\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

class EgoviewReconsRouteScale:
    def __init__(self, config_path,OPTIMIZE = False,Car_Type = "CD701",_dataset_index:int=0):
        self.Car_Type = Car_Type
        self.id_flag = True
        self.OPTIMIZE = OPTIMIZE
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract paths from config
        try:
            self.target_dataset = self.config['utdv2_settings']['target_dataset'][_dataset_index]
            self.base_path = self.target_dataset
        except KeyError as e:
            print(f"Missing required path in config: {e}")
            raise
        # Check if base path exists
        if not os.path.exists(self.base_path):
            print(f"Error: Base path {self.base_path} does not exist")
            return

        # Check if location and vision folders exist
        location_path = os.path.join(self.base_path, "location")
        vision_path = os.path.join(self.base_path, "vision")
        
        if not os.path.exists(location_path) or not os.path.exists(vision_path):
            print(f"Error: Required folders 'location' and 'vision' not found in {self.base_path}")
            return
        
        
        def get_dats_list(location_path:str,vision_path:str):
            # Check for files or folders in location directory matching the pattern
            location_matches = []
            # Check for directories in location path
            for subdir in os.listdir(location_path):
                subdir_path = os.path.join(location_path, subdir)
                if os.path.isdir(subdir_path):
                # Check files within each subdirectory
                    for item in os.listdir(subdir_path):
                        if re.match(PATTERN_701, item):
                            location_matches.append(item.replace('.csv', ''))

            # Check for files or folders in vision directory matching the pattern
            vision_matches = []
            # Check for directories in vision path
            for subdir in os.listdir(vision_path):
                subdir_path = os.path.join(vision_path, subdir)
                if os.path.isdir(subdir_path):
                # Check files within each subdirectory
                    for item in os.listdir(subdir_path):
                        if re.match(PATTERN_701, item):
                            vision_matches.append(item)

            # Find common matches between location and vision directories
            common_matches = set(location_matches).intersection(set(vision_matches))
            
            if not common_matches:
                print(f"Error: No files/folders matching CD701 pattern found in both location and vision directories")
                return
            return common_matches
        common_matches = get_dats_list(location_path, vision_path)
        # Use the first match (or iterate through all matches if needed)
        # target_name = next(iter(common_matches))

        def get_full_path(target_name):
            location_file = os.path.join(location_path, f"{target_name}.csv")
            second_vis_folder = os.listdir(vision_path)[0]
            vision_dir = os.path.join(vision_path, second_vis_folder,target_name)

            second_loc_folder = os.listdir(location_path)[0]
            location_file = os.path.join(location_path, second_loc_folder,f"{target_name}.csv")
            location_exists = os.path.exists(location_file)
            vision_exists = os.path.exists(vision_dir)
        
            if not location_exists or not vision_exists:
                print(f"Error: Required file or folder '{target_name}' not found in location or vision directories")
                return

            print(f"Validation successful: Base path and required files/folders exist for {target_name}")
            return location_exists,vision_exists
        # Check if we have valid matches
        if not common_matches:
            print(f"Error: No matching datasets found")
            return
            
        # Verify each match has both location and vision data
        valid_matches = []
        for match in common_matches:
            location_exists, vision_exists = get_full_path(match)
            if location_exists and vision_exists:
                valid_matches.append(match)
            else:
                print(f"Warning: Missing data for {match}, skipping")
            
        if not valid_matches:
            print(f"Error: No valid matches with both location and vision data")
            return
        
        # Create dictionaries to store complete paths for each valid match
        self.location_files = {}
        self.vision_dirs = {}

        # For each valid match, find and store the complete paths
        for match in valid_matches:
            # Find the subdirectory in location that contains this match
            for loc_subdir in os.listdir(location_path):
                loc_subdir_path = os.path.join(location_path, loc_subdir)
                if os.path.isdir(loc_subdir_path):
                    csv_file = os.path.join(loc_subdir_path, f"{match}.csv")
                    if os.path.exists(csv_file):
                        self.location_files[match] = csv_file
                        break
            
            # Find the subdirectory in vision that contains this match
            for vis_subdir in os.listdir(vision_path):
                vis_subdir_path = os.path.join(vision_path, vis_subdir)
                if os.path.isdir(vis_subdir_path):
                    vis_dir = os.path.join(vis_subdir_path, match)
                    if os.path.exists(vis_dir):
                        self.vision_dirs[match] = vis_dir
                        break

        print(f"Found {len(self.location_files)} location files and {len(self.vision_dirs)} vision directories")
        self.valid_matches = valid_matches
        self.camera_paras_setting()
    def camera_paras_setting(self):
        if self.Car_Type == "C385":
            self.dist_coeffs_cam0 = np.array([
                0.639999986,
                -0.0069,
                0.00065445,
                0.000117648,
                -0.0057,
                1.003900051,
                0.131899998,
                -0.020199999
            ])

            self.K_cam0 = np.array([[1907.819946, 0, 1903.890015],
                                    [0, 1907.670044, 1065.380005],
                                    [0, 0, 1]]) 
            self.cameraInfo = Info({
                "focalLengthX": int(1907.819946),   # focal length x
                "focalLengthY": int(1907.670044),   # focal length y
                "opticalCenterX": int(1903.890015), # optical center x
                "opticalCenterY": int(1065.380005), # optical center y
                "cameraHeight": 1340,    # camera height in `mm`
                "pitch": -0.030369*(180/pi),    # rotation degree around x
                "yaw": 0.028274*(180/pi),   # rotation degree around y
                "roll": -0.006632*(180/pi),  # rotation degree around z
                "k1":0.639999986,
                "k2":-0.0069,
                "p1":0.00065445,
                "p2":0.000117648,
                "k3":-0.0057,
                "k4":1.003900051,
                "k5":0.131899998,
                "k6":-0.020199999,
                "tx":1.77,
                "ty":0.07,
                "tz":1.34,
                # "rx":-0.03037,
                # "ry":0.028274,
                # "rz":-0.006632
                # "rx":-0.03037-pi/2,
                # "ry":0.028274,
                # "rz":-0.006632-pi/2
                "rx":-0.03037,
                "ry":0.028274+pi/2,
                "rz":-0.006632-pi/2
            })
        elif self.Car_Type == "CD701":
            self.dist_coeffs_cam0 = np.array([0.272893995,
                                            -0.138833001,
                                            -0.000043,
                                            0.000081,
                                            -0.008432,
                                            0.634904981,
                                            -0.129685998,
                                            -0.042598002,
                                            ])
            
            self.K_cam0 = np.array([[1908.79, 0, 1926.01],
                                    [0, 1908.71, 1075.9],
                                    [0, 0, 1]])

            self.cameraInfo = Info({
                "focalLengthX": int(1908.79),   # focal length x #沿用C385
                "focalLengthY": int(1908.71),   # focal length y 沿用C385
                "opticalCenterX": int(1926.01), # optical center x 沿用C385
                "opticalCenterY": int(1075.9), # optical center y 沿用C385
                "cameraHeight": 1480,    # camera height in `mm`  改动，使用"valid_height"字段值但存疑
                "pitch": -0.011868*(180/pi),    # rotation degree around x  改动，使用新pitch值
                "yaw": 0.003491*(180/pi),   # rotation degree around y 改动，使用新yaw值
                "roll": 0.002793*(180/pi),  # rotation degree around z 改动，使用新roll值
                "k1":0.272893995, # 畸变系数，基本沿用C385
                "k2":-0.138833001, # 畸变系数，基本沿用C385
                "p1":-0.000043, # 畸变系数，基本沿用C385
                "p2":0.000081, # 畸变系数，基本沿用C385
                "k3":-0.008432, # 畸变系数，基本沿用C385
                "k4":0.634904981, # 畸变系数，基本沿用C385
                "k5":-0.129685998, # 畸变系数，基本沿用C385 
                "k6":-0.042598002, # 畸变系数，基本沿用C385
                "tx":2.26,
                "ty":-0.001381,
                "tz":1.48,
                "rx":-0.011868,
                "ry":0.003491+pi/2,
                "rz":0.002793-pi/2
            })

        self.six_dof_data = np.array([self.cameraInfo.tx, 
                                 self.cameraInfo.ty, 
                                 self.cameraInfo.tz, 
                                 self.cameraInfo.rx, 
                                 self.cameraInfo.ry, 
                                 self.cameraInfo.rz])
        
        rot_mat_ca,trans_vec_ca = self.get_camera_pose()
        self.extrinsic_matrix = camera_pose_to_extrinsic(rot_mat_ca,trans_vec_ca)
        print("extrinsic_matrix:\n",self.extrinsic_matrix)
        self.extrinsic_rotation_matrix = self.extrinsic_matrix[:3,:3]
        self.extrinsic_transaction_vector = self.extrinsic_matrix[:3,3]

    def get_camera_pose(self):
        rot_ca = sciR.from_euler('zyx',[
                                self.cameraInfo.rz,\
                                self.cameraInfo.ry,\
                                self.cameraInfo.rx
                                ])
        trans_ca = [
            self.cameraInfo.tx,
            self.cameraInfo.ty,
            self.cameraInfo.tz
        ]
        trans_vec_ca = np.array(trans_ca)
        rot_mat_ca = rot_ca.as_matrix()
        self.pose_rotation_matrix = rot_mat_ca
        self.pose_transaction_vector = trans_vec_ca
        pose_matrix = np.hstack((rot_mat_ca, \
                                 trans_vec_ca.reshape(3, 1)))
        return rot_mat_ca,trans_vec_ca
    
    def batch_recons_for_single_rao_dat(self):
        # def batch_recons_for_single_rao_dat(self):
        total_matches = len(self.valid_matches)
        for idx, match in enumerate(self.valid_matches):
            progress = (idx + 1) / total_matches * 100
            print(f"******Processing {match}... [{idx+1}/{total_matches}] ({progress:.1f}%)******")
            location_file = self.location_files[match]
            vision_dir = self.vision_dirs[match]
                # Call the reconstruction method for each match
            self.recons_for_single_rao_dat(location_file, vision_dir,match)
    # @staticmethod
    def recons_for_single_rao_dat(self,location_file,vision_dir,dat_name):
        print(f"Reconstructing for {location_file} and {vision_dir}...")

        #第一步，批量执行语义分割，将目标目录中的所有图片进行分割提取
        run_segmentation(vision_dir)
        
        #第二步，执行定位和矢量关系
        _current_sample = DataSamplePathWrapper(
            loc_path=location_file,
            vis_path=vision_dir,
            dat_name=dat_name,
        )

        target_processing_folder = os.path.join(self.target_dataset,'recons_preprocessing')
        # Check if target folder exists, create if it doesn't
        self.target_processing_folder = target_processing_folder
        if not os.path.exists(self.target_processing_folder):
            os.makedirs(self.target_processing_folder, exist_ok=True)
            print(f"Created directory: {self.target_processing_folder}")

        _current_sample.write_sample_to_target_folder(self.target_processing_folder,self.id_flag,_current_sample.dat_name)

        #第三步，执行重建
        target_result_folder = os.path.join(self.target_dataset,'recons_result')
        self.target_result_folder = target_result_folder
        self.main_reconstruction(_current_sample)


    def main_reconstruction(self,_current_sample:DataSamplePathWrapper,mask_mode:str="pkl"):
        dat_name = _current_sample.dat_name
        target_dir = self.target_result_folder
        _current_slice_output_folder = os.path.join(target_dir, dat_name)

        # 获取分割结果路径
        temp_data_root = os.path.dirname(os.path.dirname(_current_slice_output_folder))
        # Create path to the vision directory
        vision_path = os.path.join(temp_data_root, "vision")
        if os.path.exists(vision_path):
            # Get the list of subdirectories
            subdirs = [d for d in os.listdir(vision_path) if os.path.isdir(os.path.join(vision_path, d))]
            if subdirs:
                # Assuming there's only one subdirectory
                vision_subfolder = os.path.join(vision_path, subdirs[0])
                if_already_trans = True
            else:
                print(f"No subdirectories found in {vision_path}")
                return -1
        else:
            print(f"Vision directory does not exist at {vision_path}")
            return -1
        if mask_mode=="pkl":
            temp_mask_path = os.path.join(vision_subfolder, dat_name+'_seg')
            if os.path.exists(temp_mask_path):
                print(f"Mask directory exists at {temp_mask_path}")
        
        temp_loc_file = _current_sample.loc2vis_path 
            
        # temp_traj_file = glob.glob(os.path.join(temp_data_root, "trajectory*"))
        # geojson_files = glob.glob(os.path.join(temp_data_root, "*.geojson"))
        # geojson_files = [f for f in geojson_files if re.match(r'\d+_\d+\.geojson', os.path.basename(f))]
        # if len(temp_traj_file) > 0:
        #     temp_traj_file = temp_traj_file[0]
        # else:
        #     temp_traj_file = ''

        # if len(geojson_files) > 0:
        #     geojson_file = geojson_files[0]
        # else:
        #     geojson_file = ''
        # if self.if_need_transform and if_already_trans==False:
        #     # 因为需要重建两次，因此标注一下if_already_trans为False时候，才执行坐标转换
        #     print(f"run trans to file:{geojson_file},{temp_traj_file}")
        #     CoordProcessor.trans_gcj02towgs84_replace(temp_traj_file)
        #     CoordProcessor.trans_gcj02towgs84_replace(geojson_file)
        traj_correction_dict = ins_trans_util(temp_loc_file)
        # traj_correction_dict = match_trajectory_to_insdata(temp_traj_file,\
        #                                                             temp_loc_file)
            
        self.transation_instance(
                mask_dir_path=temp_mask_path,
                location_data_path=temp_loc_file,
                traj_correction_dict=traj_correction_dict,
                output_file_name=dat_name,
                mask_mode = mask_mode,
                default_output_path=target_dir
            )

    def transation_instance(self,
                            mask_dir_path: str, # 分割得到的pkl文件目录
                            location_data_path: str, # 定位关联图片的文件（loc2vis.csv）
                            traj_correction_dict: dict,
                            output_file_name: str, # 输出结果的路径
                            mask_mode = "pkl",  #此处默认的分割类型应当是pkl二进制文件类型
                            default_output_path = "reconstruction_clip1_1109/"):
        camera_matrix = self.K_cam0
        # dist_coeffs = np.zeros(5)  # assuming no distortion
        dist_coeffs = self.dist_coeffs_cam0
        # R = np.eye(3)  # replace with actual rotation matrix from camera to vehicle
        # t = np.array([0, 0, vehicle_height])  # replace with actual translation vector
        R = self.extrinsic_rotation_matrix  # replace with actual rotation matrix from camera to vehicle
        t = self.extrinsic_transaction_vector  # replace with actual translation vector
        vehicle_height = t[-1]  # example camera height from ground

        format_str = "Camera matrix:\n{}\nDistortion coefficients: {}"
        print(format_str.format(camera_matrix, dist_coeffs))

        mask_path = str(Path(mask_dir_path).absolute())
        if os.path.isdir(mask_path):
            files = sorted(glob.glob(os.path.join(mask_path, '*.*'))) 
        else:
            return
        fixed_param = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "R": self.pose_rotation_matrix,
            "t": self.pose_transaction_vector,
            "vehicle_height": vehicle_height
        }
        features = []

        features_uncertainty = []
        #读取包含定位数据
        _loc_path = str(Path(location_data_path).absolute())
        loc_data_df = input.read_loc_data(_loc_path)
        
        for file in files:
            # 从定位数据中获取四元数、世界坐标、欧拉角
            quat,\
            world_coords_from_ins,\
            rph,\
            vel = get_averaged_pose_data(loc_data_df,\
                                                    file,\
                                                    "pic_0",\
                                                    mask_mode)
            if quat is None and world_coords_from_ins is None and rph is None and vel is None:
                continue
            if vel<-0.01:
                vel = 0.01
            rot_param = [quat, \
                        rph]
            # print("rph:",rph)
            if quat == None and world_coords_from_ins==None:
                continue
            # print(file)
            def find_closest_point(target_point, dataframe):
                """
                找到与目标点距离最近的DataFrame中的点

                参数:
                target_point : tuple
                    目标点的经纬度 (latitude, longitude)。
                dataframe : pd.DataFrame
                    包含经纬度数据的DataFrame。

                返回:
                pd.Series
                    距离目标点最近的DataFrame中的行。
                """
                # 计算目标点与DataFrame中每个点的欧几里得距离
                distances = dataframe.apply(lambda row: euclidean((row['new_longitude'], \
                                                                row['new_latitude']), \
                                                                target_point), \
                                                                axis=1)
                
                # 找到距离最近的点的索引
                closest_index = distances.idxmin()
                
                # 返回距离最近的那行数据
                return dataframe.loc[closest_index]
            
            closest_point = find_closest_point(world_coords_from_ins, \
                                               traj_correction_dict)

            if self.OPTIMIZE==True:
                # 执行优化，即采用new_longitude和new_latitude
                world_coords_from_ins=(closest_point.iloc[5],\
                                    closest_point.iloc[4])
            elif self.OPTIMIZE==False:
                # 执行非优化，即采用原始的经纬度
                world_coords_from_ins=(closest_point.iloc[3],\
                                    closest_point.iloc[2])
            # print(world_coords_from_ins)

            # 此处是从pickle数据读取分割结果，此处应当替换为图片效率更高
            if mask_mode == "jpg":
                sem_seg = read_segmentation_mask_from_image(file)
            else:
                sem_seg = read_segmentation_mask_from_pickle(file)


            instance_edge_points_list = segment_mask_to_utilized_field_mask(sem_seg,\
                                                                            fixed_param)
            # ins_seg = semantic_to_instance_segmentation(sem_seg)
            # print(instance_edge_points_list)
            
            for _edge_points_for_one_instance in instance_edge_points_list:
                # print(len(instance_edge_points_list))
                coordinates = []
                for pixel in _edge_points_for_one_instance:
                    point_camera, point_vehicle = self.pixel_to_world_new(pixel[0],\
                                                                        pixel[1], \
                                                                        camera_matrix, \
                                                                        dist_coeffs, \
                                                                        self.pose_rotation_matrix, \
                                                                        self.pose_transaction_vector, \
                                                                        vehicle_height)
                
                    _fx = camera_matrix[0][0]
                    _fy = camera_matrix[1][1]
                    _f = math.sqrt(_fx*_fy)
                    [_x,_y,_z] = point_camera

                    _pdop_value = UncertaintyCalculator.calculate_PDOP(
                                                                        _f, \
                                                                        _z, \
                                                                        _x, \
                                                                        _y, \
                                                                        10, \
                                                                        vel/30
                                                                        )
                    # _l = _velocity / _frame_rate

                    # pdop_value = UncertaintyCalculator.calculate_pdop(_f,\
                    #                                                   )
                    # 将自车坐标转化为世界坐标
                    point_world = trans_ego_to_world_coord(point_vehicle = point_vehicle, 
                                                           quanternion = rot_param, 
                                                           geographical_coords=world_coords_from_ins)

                    # q,world_coords_from_ins
                    # coordinates.append((point_vehicle[0], point_vehicle[1],0))  # 添加点坐标到坐标列表中
                    # coordinates.append((point_world[0], point_world[1],0.0))  # 添加点坐标到坐标列表中
                    coordinates.append((point_world[0], point_world[1]))  # 添加点坐标到坐标列表中
                    
                    feat_uncertainty = geojson.Feature(
                        geometry=geojson.Point([float(point_world[0]), float(point_world[1])]),
                        properties={"uncertainty": float(_pdop_value)}
                    )

                    features_uncertainty.append(feat_uncertainty)
                    # print(
                    #     "point_camera:",\
                    #     point_camera,\
                    #     "\npoint_vehicle:",\
                    #     point_vehicle,"\n",
                    #     "point_world:",\
                    #     point_world)
                # coordinates = simplify_polygon(coordinates)
                
                
                feature = geojson.Feature(
                    geometry=geojson.Polygon([coordinates]),  # 使用Polygon表示该实例的边界
                    properties={"file_name": os.path.basename(file)}  # 将文件名作为属性
                )
                features.append(feature)
        feature_collection = geojson.FeatureCollection(features)
        feature_collection_uncertainty = geojson.FeatureCollection(features_uncertainty)

        if os.path.exists(default_output_path)==False:
            os.mkdir(default_output_path)

        _log = calculate_average_movement(df=traj_correction_dict,
                                   uuid=output_file_name)
        # import math
        # import json

        def clean_inf_nan(data):
            """递归遍历 JSON 结构，将 `inf` 和 `nan` 替换为 None"""
            if isinstance(data, dict):
                return {k: clean_inf_nan(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_inf_nan(v) for v in data]
            elif isinstance(data, float):
                return None if math.isinf(data) or math.isnan(data) else data
            else:
                return data

        feature_collection_uncertainty = clean_inf_nan(feature_collection_uncertainty)
        # 将重建结果写入文件
        write_reconstructed_result(
            default_output_path,\
            output_file_name,\
            feature_collection_uncertainty,\
            recons_log = _log,\
            IF_OPTIMIZE=self.OPTIMIZE)
        # with open(os.path.join(default_output_path, output_file_name+".geojson"), 'w') as f:
        #     geojson.dump(feature_collection, f)
  
        return

    @staticmethod
    def pixel_to_world_new(u, v, camera_matrix, dist_coeffs, R, T, vehicle_height):
        # 将像素坐标转化为图像坐标
        uv = np.array([[u, v]], dtype=np.float32)

        # 去畸变并归一化
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs, P=camera_matrix)
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs)
        # tips此处P参数如果不赋值，该该函数返回的结果

        # 归一化相机坐标系
        u_n, v_n = uv_undistorted[0][0]

        # 形成归一化的相机坐标
        normalized_camera_coords = np.array([u_n, v_n, 1.0])

        # 计算比例因子，假设平面高度Z=0,即每个像素换算为世界坐标系对应的距离，以米为单位
        # 这里，直接利用外参进行变换之前计算比例因子是关键步骤
        scale_factor = vehicle_height / (R[2, 0] * normalized_camera_coords[0] + 
                                        R[2, 1] * normalized_camera_coords[1] + 
                                        R[2, 2])
        # scale_factor = vehicle_height / np.dot(R[2], normalized_camera_coords)

        # 乘以比例因子得到相机坐标系中的点
        camera_coords_scaled = normalized_camera_coords * scale_factor

        # 应用外参变换，将相机坐标系坐标转换到世界坐标系
        world_coords = np.dot(R, camera_coords_scaled) + T

        # print("camera:",camera_coords_scaled, \
        #         "world:",world_coords)
        # 返回世界坐标
        return camera_coords_scaled,world_coords
        # return camera_coords_scaled,world_coords[:2]  # 通常假设z=0，返回x和y坐标

@dataclass
class EgoviewReconstruction:
    def __init__(self, config_path,OPTIMIZE = True,Car_Type = "C385"):
        self.OPTIMIZE = OPTIMIZE
        self.config_path = config_path
        # Load config from yaml file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract paths from config
        try:
            # self.array_mask_path = self.config['reconstruction_setting']['target_seg_files_path']
            self.array_mask_path = self.config['wrap_setting']['target_folder']
            self.if_need_transform = self.config['reconstruction_setting']['if_need_transform']
            # self.processed_dat_path = self.config['paths']['processed_dat_path'] 
            # self.loc_data_path = self.config['paths']['location_data_path']
        except KeyError as e:
            print(f"Missing required path in config: {e}")
            raise
        # 假设这些参数是从配置或文件中读取的
        if Car_Type == "C385":
            self.dist_coeffs_cam0 = np.array([
                0.639999986,
                -0.0069,
                0.00065445,
                0.000117648,
                -0.0057,
                1.003900051,
                0.131899998,
                -0.020199999
            ])

            self.K_cam0 = np.array([[1907.819946, 0, 1903.890015],
                                    [0, 1907.670044, 1065.380005],
                                    [0, 0, 1]]) 
            self.cameraInfo = Info({
                "focalLengthX": int(1907.819946),   # focal length x
                "focalLengthY": int(1907.670044),   # focal length y
                "opticalCenterX": int(1903.890015), # optical center x
                "opticalCenterY": int(1065.380005), # optical center y
                "cameraHeight": 1340,    # camera height in `mm`
                "pitch": -0.030369*(180/pi),    # rotation degree around x
                "yaw": 0.028274*(180/pi),   # rotation degree around y
                "roll": -0.006632*(180/pi),  # rotation degree around z
                "k1":0.639999986,
                "k2":-0.0069,
                "p1":0.00065445,
                "p2":0.000117648,
                "k3":-0.0057,
                "k4":1.003900051,
                "k5":0.131899998,
                "k6":-0.020199999,
                "tx":1.77,
                "ty":0.07,
                "tz":1.34,
                # "rx":-0.03037,
                # "ry":0.028274,
                # "rz":-0.006632
                # "rx":-0.03037-pi/2,
                # "ry":0.028274,
                # "rz":-0.006632-pi/2
                "rx":-0.03037,
                "ry":0.028274+pi/2,
                "rz":-0.006632-pi/2
            })
        elif Car_Type == "CD701":
            '''
            CD701_cam0_info = {
                "Calibration_Time":"23-09-25-11:20:03",
                "camera_x":0.0,
                "camera_y":0.0,
                "camera_z":1.4800000190734863,
                "center_u":1903.890015,
                "center_v":1065.380005,
                "distort":[0.64,-0.0069,0.000654,0.000118,-0.0057,1.0039,0.1319,-0.02019999921321869],
                "focal_u":1907.819946,
                "focal_v":1907.670044,
                "fov":120.0,
                "homegrahpy":
                [1.034273624420166,0.013729095458984375,-11.037109375,\
                -0.008372306823730469,1.033010482788086,134.24853515625,\
                    9.313225746154785e-10,-3.725290298461914e-09,0.9999980926513672],
                "image_height":2160,
                "image_width":3840,
                "intern":1,
                "loc":0,
                "mask":3,
                "pitch":-0.0015706592239439487,
                "roll":0.0038397256284952164,
                "type":0,
                "valid_height":[1576,1576],
                "vcs":{"rotation":[0.0,0.0,0.0],
                    "translation":[3.106,1.143,0.0]},
                "vendor":"ar0820",
                "version":1,
                "yaw":0.008028507232666016
            }'
            '''
            # self.dist_coeffs_cam0 = np.array([0.64,-0.0069,0.000654,0.000118,-0.0057,1.0039,0.1319,-0.02019999921321869])
            
            
            # 内参未发生变化
            # self.K_cam0 = np.array([[1907.819946, 0, 1903.890015],
            #                         [0, 1907.670044, 1065.380005],
            #                         [0, 0, 1]])
            self.dist_coeffs_cam0 = np.array([0.272893995,
                                            -0.138833001,
                                            -0.000043,
                                            0.000081,
                                            -0.008432,
                                            0.634904981,
                                            -0.129685998,
                                            -0.042598002,
                                            ])
            
            self.K_cam0 = np.array([[1908.79, 0, 1926.01],
                                    [0, 1908.71, 1075.9],
                                    [0, 0, 1]])

            self.cameraInfo = Info({
                "focalLengthX": int(1908.79),   # focal length x #沿用C385
                "focalLengthY": int(1908.71),   # focal length y 沿用C385
                "opticalCenterX": int(1926.01), # optical center x 沿用C385
                "opticalCenterY": int(1075.9), # optical center y 沿用C385
                "cameraHeight": 1480,    # camera height in `mm`  改动，使用"valid_height"字段值但存疑
                "pitch": -0.011868*(180/pi),    # rotation degree around x  改动，使用新pitch值
                "yaw": 0.003491*(180/pi),   # rotation degree around y 改动，使用新yaw值
                "roll": 0.002793*(180/pi),  # rotation degree around z 改动，使用新roll值
                "k1":0.272893995, # 畸变系数，基本沿用C385
                "k2":-0.138833001, # 畸变系数，基本沿用C385
                "p1":-0.000043, # 畸变系数，基本沿用C385
                "p2":0.000081, # 畸变系数，基本沿用C385
                "k3":-0.008432, # 畸变系数，基本沿用C385
                "k4":0.634904981, # 畸变系数，基本沿用C385
                "k5":-0.129685998, # 畸变系数，基本沿用C385 
                "k6":-0.042598002, # 畸变系数，基本沿用C385
                "tx":2.26,
                "ty":-0.001381,
                "tz":1.48,
                "rx":-0.011868,
                "ry":0.003491+pi/2,
                "rz":0.002793-pi/2
            })

        self.six_dof_data = np.array([self.cameraInfo.tx, 
                                 self.cameraInfo.ty, 
                                 self.cameraInfo.tz, 
                                 self.cameraInfo.rx, 
                                 self.cameraInfo.ry, 
                                 self.cameraInfo.rz])
        
        rot_mat_ca,trans_vec_ca = self.get_camera_pose()
        self.extrinsic_matrix = camera_pose_to_extrinsic(rot_mat_ca,trans_vec_ca)
        print("extrinsic_matrix:\n",self.extrinsic_matrix)
        self.extrinsic_rotation_matrix = self.extrinsic_matrix[:3,:3]
        self.extrinsic_transaction_vector = self.extrinsic_matrix[:3,3]
        # self.extrinsic_rotation_matrix = 
        # self.R_vec,self.T_vec,self.extrinsic_matrix = self.from_6DoF_to_Rvec(self.six_dof_data)
    def get_camera_pose(self):
        # rot_ca = sciR.from_euler('zyx',[
        #                         self.cameraInfo["rz"],\
        #                         self.cameraInfo["ry"],\
        #                         self.cameraInfo["rx"]
        #                         ])
        # trans_ca = [
        #     self.cameraInfo["tx"],
        #     self.cameraInfo["ty"],
        #     self.cameraInfo["tz"]
        # ]
        rot_ca = sciR.from_euler('zyx',[
                                self.cameraInfo.rz,\
                                self.cameraInfo.ry,\
                                self.cameraInfo.rx
                                ])
        trans_ca = [
            self.cameraInfo.tx,
            self.cameraInfo.ty,
            self.cameraInfo.tz
        ]
        trans_vec_ca = np.array(trans_ca)
        rot_mat_ca = rot_ca.as_matrix()
        self.pose_rotation_matrix = rot_mat_ca
        self.pose_transaction_vector = trans_vec_ca
        pose_matrix = np.hstack((rot_mat_ca, \
                                 trans_vec_ca.reshape(3, 1)))
        return rot_mat_ca,trans_vec_ca
    
    def from_6DoF_to_Rvec(self, six_dof_data):
        R_vec = np.array(six_dof_data[3:])
        T_vec = np.array(six_dof_data[:3])

        # Converts a rotation matrix to a rotation vector or vice versa.
        print(R_vec)
        R_matrix, _ = cv2.Rodrigues(R_vec)
        print("相机坐标系相对于世界坐标系的旋转矩阵：")
        print(R_matrix)
        pose_matrix = np.hstack((R_matrix, T_vec.reshape(3, 1)))
        print("相机位姿矩阵：")
        print(pose_matrix)

        # 由计算外参矩阵
        # extrinsic_matrix = pose_to_extrinsic(pose_matrix)
        extrinsic_matrix_rt = camera_pose_to_extrinsic(R_matrix,T_vec)
        print(f"自车坐标系原点在相机坐标系的x，y，z向量坐标：\n{extrinsic_matrix_rt @ (np.array([[0,0,0,1]]).T)}")
        return R_matrix,T_vec,extrinsic_matrix_rt#pose_matrix
    
    def get_undistort_img(self, temp='ca_cam0_sample'):
        image = cv2.imread(f"{temp}.jpg")
        if image is None:
            raise FileNotFoundError(f"Image {temp}.jpg not found.")
        height, width = image.shape[:2]

        # 根据需求选择是否执行逆透视变换（当前未使用）
        # points = [[100, 100], [200, 200]]
        # points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        # undistorted_points = cv2.undistortPoints(points, self.K_cam0, None, R=extrinsic_matrix[:3, :3], P=self.K_cam0)

        # 这里执行图像校正
        h, w = image.shape[:2]
        new_camera_matrix, \
                roi = cv2.getOptimalNewCameraMatrix(self.K_cam0, \
                                                    self.dist_coeffs_cam0, \
                                                    (w, h), \
                                                    1, \
                                                    (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(self.K_cam0, \
                                                self.dist_coeffs_cam0, \
                                                None, \
                                                new_camera_matrix, \
                                                (w, h), \
                                                5)
        
        undistorted_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
        # 可以根据需要决定是否展示或保存图像
        # cv2.imshow(temp+'_undist.jpg', undistorted_img)
        # cv2.waitKey()
        cv2.imwrite(temp+'_undist.jpg', undistorted_img)

        return undistorted_img

    @staticmethod
    def pixel_to_world(u, v, K, dist_coeffs, R, T, vehicle_height):
        # 步骤1：使用内参矩阵和畸变系数对像素坐标进行去畸变，得到校正后的像素坐标。
        undistorted_points = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), K, dist_coeffs)

        # 步骤2：将校正后的像素坐标转换为归一化相机坐标。
        # 归一化坐标即是从像素坐标除以焦距并减去主点偏移。
        normalized_camera_coords = np.array([undistorted_points[0][0][0], undistorted_points[0][0][1], 1.0])

        # 步骤3：将归一化相机坐标通过逆旋转矩阵转换为相机坐标。
        # 使用相机旋转矩阵的逆（转置）将方向从相机坐标变换到世界坐标。
        cam_to_world_rotation = np.linalg.inv(R)
        cam_coords = cam_to_world_rotation.dot(normalized_camera_coords)

        # 步骤4：假设地面水平（Z=0），计算相机坐标系中某一点与地面的交点。
        # 这里我们假设相机高度已知，并使用此来找到尺度因子。
        scale_factor = vehicle_height / cam_coords[1]  # 取Y轴高度除以相机高度

        # 计算世界坐标，使用尺度因子。
        world_coords = scale_factor * cam_coords

        # 加上平移向量，得到最终的世界坐标。
        world_coords += T

        # 返回世界坐标，Z坐标在此为0。
        return world_coords
        
    @staticmethod
    def pixel_to_world_new(u, v, camera_matrix, dist_coeffs, R, T, vehicle_height):
        # 将像素坐标转化为图像坐标
        uv = np.array([[u, v]], dtype=np.float32)

        # 去畸变并归一化
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs, P=camera_matrix)
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs)
        # tips此处P参数如果不赋值，该该函数返回的结果

        # 归一化相机坐标系
        u_n, v_n = uv_undistorted[0][0]

        # 形成归一化的相机坐标
        normalized_camera_coords = np.array([u_n, v_n, 1.0])

        # 计算比例因子，假设平面高度Z=0,即每个像素换算为世界坐标系对应的距离，以米为单位
        # 这里，直接利用外参进行变换之前计算比例因子是关键步骤
        scale_factor = vehicle_height / (R[2, 0] * normalized_camera_coords[0] + 
                                        R[2, 1] * normalized_camera_coords[1] + 
                                        R[2, 2])
        # scale_factor = vehicle_height / np.dot(R[2], normalized_camera_coords)

        # 乘以比例因子得到相机坐标系中的点
        camera_coords_scaled = normalized_camera_coords * scale_factor

        # 应用外参变换，将相机坐标系坐标转换到世界坐标系
        world_coords = np.dot(R, camera_coords_scaled) + T

        # print("camera:",camera_coords_scaled, \
        #         "world:",world_coords)
        # 返回世界坐标
        return camera_coords_scaled,world_coords
        # return camera_coords_scaled,world_coords[:2]  # 通常假设z=0，返回x和y坐标
    
    @staticmethod
    def image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height):
        # 去畸变像素点（如果使用的是畸变图像点）
        uv = np.array([[[u, v]]], dtype=np.float32)
        uv_undistorted = cv2.undistortPoints(uv, 
                                             camera_matrix, 
                                             dist_coeffs, 
                                             P=camera_matrix)
        undistorted_points = cv2.undistortPoints(uv, \
                                                camera_matrix, \
                                                dist_coeffs,\
                                                P=camera_matrix)

        # 将像素坐标转换为归一化相机坐标
        u_n = uv_undistorted[0, 0, 0]
        v_n = uv_undistorted[0, 0, 1]

        # 转换为齐次坐标
        normalized_camera_coords = np.array([u_n, v_n, 1.0])

        # 假设地面为平面，求比例因子使点落在 Z = 0 上
        # 如果知道 vehicle_height（相机到地面的高度），用它来求比例因子
        scale_factor = vehicle_height / np.dot(R[1], normalized_camera_coords)

        # 计算在相机坐标系中的3D位置
        point_camera = scale_factor * normalized_camera_coords

        # 将相机坐标转换为车辆坐标
        point_vehicle = np.dot(R, point_camera) + t

        return point_camera,point_vehicle
    
    # @print_run_time('完成1切片的重建')
    def transation_instance(self,
                            mask_dir_path: str,
                            location_data_path: str,
                            traj_correction_dict: dict,
                            output_file_name: str,
                            mask_mode = "pkl",  #此处默认的分割类型应当是pkl二进制文件类型
                            default_output_path = "reconstruction_clip1_1109/"):
        camera_matrix = self.K_cam0
        # dist_coeffs = np.zeros(5)  # assuming no distortion
        dist_coeffs = self.dist_coeffs_cam0
        # R = np.eye(3)  # replace with actual rotation matrix from camera to vehicle
        # t = np.array([0, 0, vehicle_height])  # replace with actual translation vector
        R = self.extrinsic_rotation_matrix  # replace with actual rotation matrix from camera to vehicle
        t = self.extrinsic_transaction_vector  # replace with actual translation vector
        vehicle_height = t[-1]  # example camera height from ground

        format_str = "Camera matrix:\n{}\nDistortion coefficients: {}"
        print(format_str.format(camera_matrix, dist_coeffs))

        mask_path = str(Path(mask_dir_path).absolute())
        if os.path.isdir(mask_path):
            files = sorted(glob.glob(os.path.join(mask_path, '*.*'))) 
        else:
            return
        fixed_param = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "R": self.pose_rotation_matrix,
            "t": self.pose_transaction_vector,
            "vehicle_height": vehicle_height
        }
        features = []

        features_uncertainty = []
        #读取包含定位数据
        _loc_path = str(Path(location_data_path).absolute())
        loc_data_df = m_input.read_loc_data(_loc_path)
        
        for file in files:
            # 从定位数据中获取四元数、世界坐标、欧拉角
            quat,\
            world_coords_from_ins,\
            rph,\
            vel = get_averaged_pose_data(loc_data_df,\
                                                    file,\
                                                    "pic_0",\
                                                    mask_mode)
            if vel<-0.01:
                vel = 0.01
            rot_param = [quat, \
                        rph]
            # print("rph:",rph)
            if quat == None and world_coords_from_ins==None:
                continue
            # print(file)
            def find_closest_point(target_point, dataframe):
                """
                找到与目标点距离最近的DataFrame中的点

                参数:
                target_point : tuple
                    目标点的经纬度 (latitude, longitude)。
                dataframe : pd.DataFrame
                    包含经纬度数据的DataFrame。

                返回:
                pd.Series
                    距离目标点最近的DataFrame中的行。
                """
                # 计算目标点与DataFrame中每个点的欧几里得距离
                distances = dataframe.apply(lambda row: euclidean((row['new_longitude'], \
                                                                row['new_latitude']), \
                                                                target_point), \
                                                                axis=1)
                
                # 找到距离最近的点的索引
                closest_index = distances.idxmin()
                
                # 返回距离最近的那行数据
                return dataframe.loc[closest_index]
            
            closest_point = find_closest_point(world_coords_from_ins, \
                                               traj_correction_dict)
            # print(closest_point)
            # print(world_coords_from_ins)
            if self.OPTIMIZE==True:
                world_coords_from_ins=(closest_point.iloc[5],\
                                    closest_point.iloc[4])
            elif self.OPTIMIZE==False:
                world_coords_from_ins=(closest_point.iloc[3],\
                                    closest_point.iloc[2])
            # print(world_coords_from_ins)

            # 此处是从pickle数据读取分割结果，此处应当替换为图片效率更高
            if mask_mode == "jpg":
                sem_seg = read_segmentation_mask_from_image(file)
            else:
                sem_seg = read_segmentation_mask_from_pickle(file)


            instance_edge_points_list = segment_mask_to_utilized_field_mask(sem_seg,\
                                                                            fixed_param)
            # ins_seg = semantic_to_instance_segmentation(sem_seg)
            # print(instance_edge_points_list)
            
            for _edge_points_for_one_instance in instance_edge_points_list:
                # print(len(instance_edge_points_list))
                coordinates = []
                for pixel in _edge_points_for_one_instance:
                    point_camera, point_vehicle = self.pixel_to_world_new(pixel[0],\
                                                                        pixel[1], \
                                                                        camera_matrix, \
                                                                        dist_coeffs, \
                                                                        self.pose_rotation_matrix, \
                                                                        self.pose_transaction_vector, \
                                                                        vehicle_height)
                    # def calculate_PDOP(_f, _z, _x, _y, _n, _l):
                    """
                    1、交付工作
                    专利
                    代码
                    工程化

                    webgis
                    
                    2、第二项，提前给
                    第二项，结合补实验
                    论文有一个基本的框架
                    填内容


                    3、现势性
                    4、

                    计算PDOP（Position Dilution of Precision）
                    参数:
                    a, b, c: 用于计算PDOP的系数
                    f: 焦距
                    z: 深度
                    x, y: 图像坐标
                    n: 求和的上限
                    l: 每帧的位移量

                    返回:
                    PDOP值
                    # """
                    _fx = camera_matrix[0][0]
                    _fy = camera_matrix[1][1]
                    _f = math.sqrt(_fx*_fy)
                    [_x,_y,_z] = point_camera

                    _pdop_value = UncertaintyCalculator.calculate_PDOP(
                                                                        _f, \
                                                                        _z, \
                                                                        _x, \
                                                                        _y, \
                                                                        10, \
                                                                        vel/30
                                                                        )
                    # _l = _velocity / _frame_rate

                    # pdop_value = UncertaintyCalculator.calculate_pdop(_f,\
                    #                                                   )
                    # 将自车坐标转化为世界坐标
                    point_world = trans_ego_to_world_coord(point_vehicle = point_vehicle, 
                                                           quanternion = rot_param, 
                                                           geographical_coords=world_coords_from_ins)
                    # print(point_world)
                    # q,world_coords_from_ins
                    # coordinates.append((point_vehicle[0], point_vehicle[1],0))  # 添加点坐标到坐标列表中
                    # coordinates.append((point_world[0], point_world[1],0.0))  # 添加点坐标到坐标列表中
                    coordinates.append((point_world[0], point_world[1]))  # 添加点坐标到坐标列表中
                    
                    feat_uncertainty = geojson.Feature(
                        geometry=geojson.Point([float(point_world[0]), float(point_world[1])]),
                        properties={"uncertainty": float(_pdop_value)}
                    )

                    features_uncertainty.append(feat_uncertainty)
                    # print(
                    #     "point_camera:",\
                    #     point_camera,\
                    #     "\npoint_vehicle:",\
                    #     point_vehicle,"\n",
                    #     "point_world:",\
                    #     point_world)
                # coordinates = simplify_polygon(coordinates)
                
                
                feature = geojson.Feature(
                    geometry=geojson.Polygon([coordinates]),  # 使用Polygon表示该实例的边界
                    properties={"file_name": os.path.basename(file)}  # 将文件名作为属性
                )
                features.append(feature)
        feature_collection = geojson.FeatureCollection(features)
        feature_collection_uncertainty = geojson.FeatureCollection(features_uncertainty)

        if os.path.exists(default_output_path)==False:
            os.mkdir(default_output_path)

        _log = calculate_average_movement(df=traj_correction_dict,
                                   uuid=output_file_name)
        # import math
        # import json

        def clean_inf_nan(data):
            """递归遍历 JSON 结构，将 `inf` 和 `nan` 替换为 None"""
            if isinstance(data, dict):
                return {k: clean_inf_nan(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_inf_nan(v) for v in data]
            elif isinstance(data, float):
                return None if math.isinf(data) or math.isnan(data) else data
            else:
                return data

        feature_collection_uncertainty = clean_inf_nan(feature_collection_uncertainty)
        # 将重建结果写入文件
        write_reconstructed_result(
            default_output_path,\
            output_file_name,\
            feature_collection_uncertainty,\
            recons_log = _log,\
            IF_OPTIMIZE=self.OPTIMIZE)
        # with open(os.path.join(default_output_path, output_file_name+".geojson"), 'w') as f:
        #     geojson.dump(feature_collection, f)
  
        return
        # 示例使用，输入参数需根据实际摄像机参数调整
        from .transformation_utils import SAMPLE_POINTS_IN_PIXEL as samples
        for item in samples.items():
            u, v = item[1]
            print(item[0],u, v, "")
            point_camera, point_vehicle = self.pixel_to_world_new(u, 
                                                                  v,
                                                                  camera_matrix, 
                                                                  dist_coeffs, 
                                                                  self.pose_rotation_matrix, 
                                                                  self.pose_transaction_vector, 
                                                                  vehicle_height)
            # point_camera, point_vehicle = self.image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
            print("point_camera:",point_camera,"\npoint_vehicle:", point_vehicle,"\n")
        u, v = 1442, 1463  # example pixel coordinates
        vehicle_height = self.cameraInfo.tz # 车辆高度（单位：米）
        point_vehicle = self.image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        print("Vehicle coordinates:", point_vehicle)

        # 示例使用，输入参数需根据实际摄像机参数调整

        # 计算世界坐标
        world_point = self.pixel_to_world(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        world_point_new = self.pixel_to_world_new(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        print("世界坐标:", world_point)
    # @print_run_time("100clips重建")
    def batch_ego_reconstruction(self,
                                 target_dir="",
                                 mask_mode="pkl",
                                 if_already_trans=False):
        target_dir = self.array_mask_path
        from concurrent.futures import ThreadPoolExecutor 
        """
        批量进行ego数据重建
        Args:
            target_dir (str, optional): 目标文件夹路径，默认为"output"。

        Returns:
            None
        """
        def process_file(subfolder_in_uuid, target_dir):
            print(subfolder_in_uuid)
            _current_slice_output_folder = os.path.join(target_dir, subfolder_in_uuid)
            # Check if geojson file already exists
            # if os.path.exists(os.path.join(_current_slice_output_folder, f"{subfolder_in_uuid}.geojson")):
            #     print(f"当前文件夹已包含重建点集: {subfolder_in_uuid}")
            #     return
            temp_data_root = os.path.join(target_dir, subfolder_in_uuid)
            if mask_mode=="pkl":
                temp_mask_path = os.path.join(temp_data_root, "array_mask")
            elif mask_mode=="jpg":
                temp_mask_path = os.path.join(temp_data_root, "image_mask")
            temp_loc_file = os.path.join(temp_data_root, "loc2vis.csv")
            
            temp_traj_file = glob.glob(os.path.join(temp_data_root, "trajectory*"))
            geojson_files = glob.glob(os.path.join(temp_data_root, "*.geojson"))
            geojson_files = [f for f in geojson_files if re.match(r'\d+_\d+\.geojson', os.path.basename(f))]
            if len(temp_traj_file) > 0:
                temp_traj_file = temp_traj_file[0]
            else:
                temp_traj_file = ''

            if len(geojson_files) > 0:
                geojson_file = geojson_files[0]
            else:
                geojson_file = ''
            if self.if_need_transform and if_already_trans==False:
                # 因为需要重建两次，因此标注一下if_already_trans为False时候，才执行坐标转换
                print(f"run trans to file:{geojson_file},{temp_traj_file}")
                CoordProcessor.trans_gcj02towgs84_replace(temp_traj_file)
                CoordProcessor.trans_gcj02towgs84_replace(geojson_file)

            traj_correction_dict = match_trajectory_to_insdata(temp_traj_file,\
                                                                    temp_loc_file)
            
            self.transation_instance(
                mask_dir_path=temp_mask_path,
                location_data_path=temp_loc_file,
                traj_correction_dict=traj_correction_dict,
                output_file_name=subfolder_in_uuid,
                mask_mode = mask_mode,
                default_output_path=_current_slice_output_folder
            )

        files = os.listdir(target_dir)
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_file, \
                                       file, \
                                       target_dir) for file in files]

        # 等待所有任务完成
        for future in futures:
            future.result()
        
        # files = os.listdir(target_dir)
        # for file in files:
        #     print(file)
        #     temp_data_root = os.path.join(target_dir, file)
        #     if mask_mode=="pkl":
        #         temp_mask_path = os.path.join(temp_data_root, "array_mask")
        #     elif mask_mode=="jpg":
        #         temp_mask_path = os.path.join(temp_data_root, "image_mask")
        #     temp_loc_file = os.path.join(temp_data_root, "loc2vis.csv")
            
        #     temp_traj_file = glob.glob(os.path.join(temp_data_root, "trajectory*"))
        #     if len(temp_traj_file) > 0:
        #         temp_traj_file = temp_traj_file[0]
        #     else:
        #         temp_traj_file = ''

        #     traj_correction_dict = match_trajectory_to_insdata(temp_traj_file,temp_loc_file)
            
        #     self.transation_instance(
        #         DEFAULT_MASK_FILE_PATH = temp_mask_path,
        #         location_data_path = temp_loc_file,
        #         traj_correction_dict = traj_correction_dict,
        #         output_file_name=file,
        #         mask_mode = mask_mode
        #     )
        
    def inverse_perspective_mapping(self, undistorted_img):
        height = int(undistorted_img.shape[0]) # row y
        width = int(undistorted_img.shape[1]) # col x
        
        # 定义IPM参数，定义逆透视变换需要选取的图像范围
        ipm_factor = 0.6
        
        self.ipmInfo = Info({
            "inputWidth": width,
            "inputHeight": height,
            "left": 50,
            "right": width-50,
            "top": height*ipm_factor, #选取需要进行逆透视变化的图像范围
            "bottom": height-50
        })
        ipmInfo = self.ipmInfo
        cameraInfo = self.cameraInfo
        R = undistorted_img
        # vpp = GetVanishingPoint(self.cameraInfo)
        # 1.获取逆透视变换之后的特征点
        p_set = []
        self.transation_instance()

        # 2.获取逆透视变换之后的图像
        vpp = get_vanishing_point(self)
        vp_x = vpp[0][0]
        vp_y = vpp[1][0]
        print(ipmInfo.top)
        ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
        print(ipmInfo.top)
        # uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
        #         [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)
        uvLimitsp = np.array([[ipmInfo.left, ipmInfo.right, ipmInfo.right, ipmInfo.left],
                [ipmInfo.top, ipmInfo.top, ipmInfo.bottom, ipmInfo.bottom]], np.float32)
        xyLimits = TransformImage2Ground(uvLimitsp, self.cameraInfo)

        print(xyLimits)
        row1 = xyLimits[0, :]
        row2 = xyLimits[1, :]
        xfMin = min(row1)
        xfMax = max(row1)
        yfMin = min(row2)
        yfMax = max(row2)
        xyRatio = (xfMax - xfMin)/(yfMax - yfMin)
        target_height = 640
        target_width = 960
        outImage = np.zeros((target_height,target_width,4), np.float32)
        # outImage = np.zeros((640,960,4), np.float32)
        outImage[:,:,3] = 255
        # 输出图片的大小
        outRow = int(outImage.shape[0])
        outCol = int(outImage.shape[1])
        stepRow = (yfMax - yfMin)/outRow
        stepCol = (xfMax - xfMin)/outCol
        xyGrid = np.zeros((2, outRow*outCol), np.float32)
        y = yfMax-0.5*stepRow
        #构建一个地面网格（天然地平行）
        for i in range(0, outRow):
            x = xfMin+0.5*stepCol
            for j in range(0, outCol):
                xyGrid[0, (i-1)*outCol+j] = x
                xyGrid[1, (i-1)*outCol+j] = y
                x = x + stepCol
            y = y - stepRow

        #将地面格网转回图像
        # TransformGround2Image
        uvGrid = TransformGround2Image(xyGrid, cameraInfo)
        # mean value of the image
        means = np.mean(R)/255
        RR = R.astype(float)/255
        for i in range(0, outRow):
            # print(i,outRow)
            for j in range(0, outCol):
                #得到了每个点在图像中的u,v坐标
                ui = uvGrid[0, i*outCol+j]
                vi = uvGrid[1, i*outCol+j]
                #print(ui, vi)
                if ui < ipmInfo.left or ui > ipmInfo.right or vi < ipmInfo.top or vi > ipmInfo.bottom:
                    outImage[i, j] = 0.0
                else:
                    x1 = np.int32(ui)
                    x2 = np.int32(ui+0.5)
                    y1 = np.int32(vi)
                    y2 = np.int32(vi+0.5)
                    x = ui-float(x1)
                    y = vi-float(y1)
                    # print(ui, vi)
                    #双线性插值
                    outImage[i, j, 0] = float(RR[y1, x1, 0])*(1-x)*(1-y)+float(RR[y1, x2, 0])*x*(1-y)+float(RR[y2, x1, 0])*(1-x)*y+float(RR[y2, x2, 0])*x*y
                    outImage[i, j, 1] = float(RR[y1, x1, 1])*(1-x)*(1-y)+float(RR[y1, x2, 1])*x*(1-y)+float(RR[y2, x1, 1])*(1-x)*y+float(RR[y2, x2, 1])*x*y
                    outImage[i, j, 2] = float(RR[y1, x1, 2])*(1-x)*(1-y)+float(RR[y1, x2, 2])*x*(1-y)+float(RR[y2, x1, 2])*(1-x)*y+float(RR[y2, x2, 2])*x*y
        outImage[-1,:] = 0.0 
        # show the result

        outImage = outImage * 255
        print("finished")
        
        cv2.imwrite("dist_"+ipm_factor.__str__()+"_inverse_perspective_mapping_1340.jpg", outImage)

def read_segmentation_mask_from_pickle(mask_path):
    """
    从pickle文件中读取语义分割的mask图层。
    参数:
        mask_path (str): 语义分割的mask文件路径。
    返回:
        semantic_mask (numpy.ndarray): 语义分割的结果，大小为 (H, W)，
                                       每个像素值表示类别标签。
    """
    semantic_mask = m_input.read_mask_data(mask_path)
    return semantic_mask

def read_segmentation_mask_from_image(mask_path):
    """
    从图片文件中读取语义分割的mask图层。
    参数:
        mask_path (str): 语义分割的mask文件路径。
    返回:
        semantic_mask (numpy.ndarray): 语义分割的结果，大小为 (H, W)，
                                       每个像素值表示类别标签。
    """
    semantic_mask = m_input.read_mask_data(mask_path,"jpg")
    return semantic_mask

@dataclass
class UncertaintyCalculator:
    def __init__(self):
        self
    @staticmethod
    def calculate_PDOP(_f, _z, _x, _y, _n, _l):
        """
        计算PDOP（Position Dilution of Precision）
        参数:
        a, b, c: 用于计算PDOP的系数
        f: 焦距
        z: 深度
        x, y: 图像坐标
        n: 求和的上限
        l: 每帧的位移量

        返回:
        PDOP值
        # """
        # term1 = 1 / a
        # term2 = c / (a * (c - b)**2) + a / ((a * (c - b)**2) * (x**2 + y**2))
        
        # # 简化后的近似公式
        # pdop_approx = (2 * (z + n * l) * math.sqrt(z)) / (n * l * f * math.sqrt(n + 3))
        # pdop_approx += (3 * z * (z + n * l)) / ((x**2 + y**2) * math.sqrt(z + n * l))
        

        z, n, l, f = sp.symbols('z n l f')
        pdop_expression1 = 2 * (z + n * l) * sp.sqrt(z) / (n * l * f * sp.sqrt(n + 3))
        exp1_values = {
            z: _z, \
            n: _n, \
            l: _l, \
            f: _f}
        
        pdop_expression2 = sp.sqrt((3*z + n*l) + 3*z**2 * (z + n*l) / (x**2 + y**2))
        exp2_values = {
            z: _z,
            n: _n,
            l: _l,
            x: _x,
            y: _y
        }

        evaled_exp1 = pdop_expression1.subs(exp1_values)
        evaled_exp2 = pdop_expression2.subs(exp2_values)
        pdop_result = evaled_exp1.evalf() * evaled_exp2.evalf()

        # Check if pdop_result is a sympy Float
        from sympy.core.numbers import Float

        if not isinstance(pdop_result, Float):
            # Convert to float if possible, otherwise return a default value
            try:
                pdop_result = float(pdop_result)
            except (TypeError, ValueError):
                pdop_result = float('inf')  # Default value for invalid results
                                                                                                                                     
        return pdop_result

f = 1.9  # 焦距
z = 1.5  # 深度
x = 0.5  # 图像坐标x
y = 0.5  # 图像坐标y
n = 100  # 求和上限
l = 0.01  # 每帧位移量

# 创建类实例
uncertainty_calculator = UncertaintyCalculator()

# 调用PDOP计算方法
pdop_value = uncertainty_calculator.calculate_PDOP(f, z, x, y, n, l)
pdop_value
