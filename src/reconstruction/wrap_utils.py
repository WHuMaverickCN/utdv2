import re
import os
import uuid
import json
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass
from datetime import datetime,timedelta
from src.io import m_output as output
from src.io import m_input as input

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率


@dataclass
class DataSamplePathWrapper:
    loc_path:str
    vis_path:str
    dat_name:str
    vehicle_type:str="CD701"
    #此类别用于样本的打包与整理
    vec_path:str=None
    traj_path:str=None
    def __private():
        pass
    def __post_init__(self):
        self.loc_data = self.__match_vec_to_loc()
        if self.vehicle_type=="C385":
            self.loc_2_vis,self.vis_data = self.__match_vec_to_vis_TR()
        elif self.vehicle_type=="CD701":
            self.loc_2_vis,self.vis_data = self.__match_vec_to_vis_TR_CD701()

    def write_sample_to_target_folder(self,target_folder,id,dat_name):
        # 写入文件到目标文件夹
        if id == True:
            self.loc2vis_path = output.write_to_foler(dat_name,
                                self.loc_2_vis,
                                target_folder = target_folder
                            )
    
    def __match_vec_to_vis_TR(self):
        _all_pic = []
        for i in range(len(self.vis_path)):
            all_pic_files = os.listdir(self.vis_path[i])
            _all_pic = _all_pic + all_pic_files

        def re_organize_timestamp(df,target_timestamp_column_name):
            if 'timestamp' in df.columns:
                _base_time = df[target_timestamp_column_name].min()
                df['cam_pkg_time'] = df[target_timestamp_column_name] - _base_time
            elif 'utc' in df.columns:
                _base_time = df[target_timestamp_column_name].min()
                df['loc_pkg_time'] = df[target_timestamp_column_name] - _base_time
            return df
            
        def parse_pic_list(pic_list):
            data = []
            for pic in pic_list:
                match = re.match(r'(\d+\.\d+)_\d+_(\d)\.jpg', pic)
                if match:
                    timestamp = float(match.group(1))
                    camera = int(match.group(2))
                    data.append((timestamp, camera, pic))
            return pd.DataFrame(data, columns=['timestamp', 'camera', 'pic'])
        
        def find_matching_pics(loc, cam):
            merge_df = pd.merge_asof(loc.sort_values('utc'),
                          cam[['cam_pkg_time','timestamp','camera', 'pic']].sort_values('timestamp'),
                          left_on = 'loc_pkg_time',
                          right_on = 'cam_pkg_time',
                          tolerance= 0.09,
                          direction='nearest')
            
            # 修改列名
            if cam.camera.iloc[0] == 0:
                merge_df.rename(columns={'camera': 'camera_0','pic': 'pic_0'}, inplace=True)
            elif cam.camera.iloc[0] == 1:
                merge_df.rename(columns={'camera': 'camera_1','pic': 'pic_1'}, inplace=True)
            # import matplotlib.pyplot as plt
            # df = pd.DataFrame(columns=['value'])
            # df['value'] = merge_df['loc_pkg_time']-merge_df['cam_pkg_time']
            # plt.figure(figsize=(10, 6))
            # plt.plot(df.index, df['value'], marker='o', linestyle='-', color='b')
            # plt.title('Line Plot of Single Column DataFrame with Date Index')
            # plt.xlabel('Date')
            # plt.ylabel('Value')
            # plt.grid(True)
            # plt.show()

            return merge_df
        
        pic_df = parse_pic_list(_all_pic)
        pic_df = re_organize_timestamp(pic_df,'timestamp')

        ro_cam0 = pic_df[pic_df['camera']==0]
        ro_cam1 = pic_df[pic_df['camera']==1] 

        target_df_loc = input.read_sample_location_file(self.loc_path) #这里需要整体的C385数据包解出的定位数据
        target_df_loc = TimeStampProcessor.get_extra_suffix_dataframe(target_df_loc)
        ro_loc = re_organize_timestamp(target_df_loc,'utc')
        
        # result_df = find_matching_pics(target_df_loc, pic_df)
        mathced_result = find_matching_pics(ro_loc, ro_cam0) #5826×62
        print("mathced_result.shape:",mathced_result.shape)
        mathced_result = find_matching_pics(mathced_result, ro_cam1) #5826×66
        print("mathced_result.shape:",mathced_result.shape)
        
        def find_closest_utc(df, utc_value):
            diff = np.abs(df['utc'] - utc_value)
            idx = diff.idxmin()
            return df.loc[idx]

        # 使用函数查找loc_data中每个utc最近的matched_result中的行
        closest_rows = self.loc_data['utc'].apply(lambda x: find_closest_utc(mathced_result, x))
        print("closest_rows.shape:", closest_rows.shape)
        # 将找到的行拼接到loc_data中
        mathced_result = pd.concat([self.loc_data.reset_index(drop=True), closest_rows.reset_index(drop=True)], axis=1)
        print("mathced_result.shape:", mathced_result.shape)
        

        def is_valid_image_string(img):
            # 检查 img 是否是字符串，并且以 '_0.jpg' 或 '_1.jpg' 结尾
            return isinstance(img, str) and (img.endswith('_0.jpg') or img.endswith('_1.jpg'))
        # camera_0_images = [img for img in mathced_result['pic_0'].tolist() if img.endswith('_0.jpg')]
        # camera_1_images = [img for img in mathced_result['pic_1'].tolist() if img.endswith('_1.jpg')]
        camera_0_images = [img for img in mathced_result['pic_0'] if is_valid_image_string(img)]
        camera_1_images = [img for img in mathced_result['pic_1'] if is_valid_image_string(img)]
        
        camera_0_images = [os.path.join(self.vis_path[0],item) for item in camera_0_images]
        camera_1_images = [os.path.join(self.vis_path[0],item) for item in camera_1_images]
        
        # #添加反偏转
        # def convert_row(row):
        #     new_lng, new_lat = CoordProcessor.gcj02towgs84_point_level(row['longitude'], row['latitude'])
        #     return pd.Series([new_lng, new_lat], index=['new_longitude', 'new_latitude'])
        # mathced_result[['new_longitude', 'new_latitude']] = mathced_result.apply(convert_row, axis=1)
        
        return mathced_result,{'cam_0':camera_0_images,'cam_1':camera_1_images}
    def __match_vec_to_vis_TR_CD701(self):
        self.vis_path = [self.vis_path] if isinstance(self.vis_path, str) else self.vis_path

        # 在701模式中，不存在两个摄像头的情况，因此需要修改匹配逻辑
        _all_pic = []
        for i in range(len(self.vis_path)):
            all_pic_files = os.listdir(self.vis_path[i])
            _all_pic = _all_pic + all_pic_files

        def CD701_re_organize_timestamp(df,target_timestamp_column_name):
            if 'timestamp' in df.columns:
                _base_time = df[target_timestamp_column_name].min()
                df['cam_pkg_time'] = df[target_timestamp_column_name] - _base_time
            elif 'utc' in df.columns:
                _base_time = df[target_timestamp_column_name].min()
                df['loc_pkg_time'] = df[target_timestamp_column_name] - _base_time
            return df
            
        def CD701_parse_pic_list(pic_list):
            data = []
            for pic in pic_list:
                match = re.match(r'(\d+)\.jpg', pic)
                if match:
                    timestamp = float(match.group(0).split('.')[0]) / 10**6
                    camera = int(0)
                    data.append((timestamp, camera, pic))
            return pd.DataFrame(data, columns=['timestamp', 'camera', 'pic'])
        
        def find_matching_pics(loc, cam):
            merge_df = pd.merge_asof(loc.sort_values('utc'),
                          cam[['cam_pkg_time','timestamp','camera', 'pic']].sort_values('timestamp'),
                          left_on = 'loc_pkg_time',
                          right_on = 'cam_pkg_time',
                          tolerance= 0.09,
                          direction='nearest')
            
            # 修改列名
            if cam.camera.iloc[0] == 0:
                merge_df.rename(columns={'camera': 'camera_0','pic': 'pic_0'}, inplace=True)
            elif cam.camera.iloc[0] == 1:
                merge_df.rename(columns={'camera': 'camera_1','pic': 'pic_1'}, inplace=True)

            return merge_df
        
        pic_df = CD701_parse_pic_list(_all_pic)
        pic_df = CD701_re_organize_timestamp(pic_df,'timestamp')

        ro_cam0 = pic_df[pic_df['camera']==0]
        ro_cam1 = pic_df[pic_df['camera']==1] 

        target_df_loc = input.read_sample_location_file(self.loc_path) #这里需要整体的C385数据包解出的定位数据
        ro_loc = CD701_re_organize_timestamp(target_df_loc,'utc')
        
        # result_df = find_matching_pics(target_df_loc, pic_df)
        mathced_result = find_matching_pics(ro_loc, ro_cam0) # 5270*62
        print("mathced_result.shape:",mathced_result.shape)
        # mathced_result = find_matching_pics(mathced_result, ro_cam1)
        
        def find_closest_utc(df, utc_value):
            diff = np.abs(df['utc'] - utc_value)
            idx = diff.idxmin()
            return df.loc[idx]

        # 使用函数查找loc_data中每个utc最近的matched_result中的行
        closest_rows = self.loc_data['utc'].apply(lambda x: find_closest_utc(mathced_result, x))
        print("closest_rows.shape:",closest_rows.shape)

        # 将找到的行拼接到loc_data中
        mathced_result = pd.concat([self.loc_data.reset_index(drop=True), closest_rows.reset_index(drop=True)], axis=1)
        print("mathced_result.shape:",mathced_result.shape)
        def is_valid_image_string(img):
            # 检查 img 是否是字符串，并且以 '_0.jpg' 或 '_1.jpg' 结尾
            return isinstance(img, str)
        # camera_0_images = [img for img in mathced_result['pic_0'].tolist() if img.endswith('_0.jpg')]
        # camera_1_images = [img for img in mathced_result['pic_1'].tolist() if img.endswith('_1.jpg')]
        camera_0_images = [img for img in mathced_result['pic_0'] if is_valid_image_string(img)]
        # camera_1_images = [img for img in mathced_result['pic_1'] if is_valid_image_string(img)]
        
        camera_0_images = [os.path.join(self.vis_path[0],item) for item in camera_0_images]
        # camera_1_images = [os.path.join(self.vis_path[0],item) for item in camera_1_images]
        
        # #添加反偏转
        # def convert_row(row):
        #     new_lng, new_lat = CoordProcessor.gcj02towgs84_point_level(row['longitude'], row['latitude'])
        #     return pd.Series([new_lng, new_lat], index=['new_longitude', 'new_latitude'])
        # mathced_result[['new_longitude', 'new_latitude']] = mathced_result.apply(convert_row, axis=1)
        
        return mathced_result,{'cam_0':camera_0_images,'cam_1':camera_0_images}

    def __match_vec_to_vis(self):
        _all_pic = []
        for i in range(len(self.vis_path)):
            all_pic_files = os.listdir(self.vis_path[i])
            _all_pic = _all_pic + all_pic_files
        
        def filter_and_organize_images(vis_path,image_list, start_time, end_time):
            # 将时间字符串转换为时间戳
            def timestamp_to_datetime(ts):
                return datetime.fromtimestamp(float(ts))
            
            # 筛选出在时间范围内的图片列表
            filtered_images = [
                img for img in image_list
                if start_time <= timestamp_to_datetime(img.split('_')[0]) <= end_time
            ]
            
            # 根据相机编号分别组织为两个列表
            camera_0_images = [img for img in filtered_images if img.endswith('_0.jpg')]
            camera_1_images = [img for img in filtered_images if img.endswith('_1.jpg')]

            camera_0_images = [os.path.join(vis_path[0],item) for item in camera_0_images]
            camera_1_images = [os.path.join(vis_path[0],item) for item in camera_1_images]
            return camera_0_images, camera_1_images

        temp_gdf_vec = self.vec_data.copy()
        temp_gdf_vec['start_time'] = temp_gdf_vec['start_time'].apply(TimeStampProcessor.convert_timestamp)
        temp_gdf_vec['end_time'] = temp_gdf_vec['end_time'].apply(TimeStampProcessor.convert_timestamp)
        
        _str_start_time = temp_gdf_vec['start_time'].iloc[0]
        _str_end_time = temp_gdf_vec['end_time'].iloc[0]
        start_time = datetime.fromtimestamp(_str_start_time)  # 开始时间
        end_time = datetime.fromtimestamp(_str_end_time)    # 结束时间

        camera_0_images, camera_1_images = filter_and_organize_images(self.vis_path,_all_pic, start_time, end_time)

        return {'cam_0':camera_0_images,'cam_1':camera_1_images}
    def __match_vec_to_loc(self):
        # 匹配轨迹和定位数据
        filtered_points = input.read_sample_location_file(self.loc_path) #此处的self.loc_path为文件路径组成的列表，可能包含多个
        def convert_row(row):
            new_lng, new_lat = CoordProcessor.gcj02towgs84_point_level(row['longitude'], row['latitude'])
            return pd.Series([new_lng, new_lat], index=['new_longitude', 'new_latitude'])
        if self.vehicle_type=="C385":
            filtered_points[['new_longitude', 'new_latitude']] = filtered_points.apply(convert_row, axis=1)
        elif self.vehicle_type=="CD701":
            # filtered_points[['new_longitude', 'new_latitude']] = filtered_points[['longitude', 'latitude']]
            filtered_points[['new_longitude', 'new_latitude']] = filtered_points.apply(convert_row, axis=1)

        return filtered_points

class TimeStampProcessor:
    @staticmethod
    def get_extra_suffix_dataframe(_dataframe):
        # _dataframe['sec_of_week_last_three'] = _dataframe['sec_of_week'].astype(str).str[-3:].astype(float) / 1000
        # _dataframe['utc'] = _dataframe['utc'] + _dataframe['sec_of_week_last_three']

        return _dataframe
    @staticmethod
    def convert_timestamp(timestamp):
        # 转换为10位float时间戳
        if len(str(timestamp)) == 16:
            _suffix = (timestamp % 10**6)/10**6
            timestamp = timestamp // 1000000 + _suffix
        else:
            timestamp_main = str(timestamp).split('.')[0]
            _scale = pow(10,len(str(timestamp_main))-10)
            timestamp = int(timestamp_main) // _scale
        return timestamp
    def __init__(self):
        pass
    @staticmethod
    def unify_timestamp(input):
        if input is str:
            print("str")
        else:
            print(type(input))
        return 1
    
    @staticmethod
    def check_period(period):
        period.start_time,precison_start_time = TimeStampProcessor.__check_timestamp_format(period.start_time)
        period.end_time,precison_end_time = TimeStampProcessor.__check_timestamp_format(period.end_time)
        precision = precison_start_time if precison_start_time==precison_end_time else max(precison_start_time,precison_end_time)
        return period,precision
    
    @staticmethod
    def __check_timestamp_format(temporal_timestamp):
        if type(temporal_timestamp) != str:
            temporal_timestamp = temporal_timestamp.__str__()
        l = len(temporal_timestamp)
        timestamp_pattern = r'^\d{10}$'
        timestamp_pattern_16 = r'^\d{16}$'

        precision = 0
        if l == 10 and re.match(timestamp_pattern, temporal_timestamp):
            precision = 10
            return float(temporal_timestamp),precision
        elif l==16 and re.match(timestamp_pattern_16, temporal_timestamp):
            precision = 16
            time_stamp_int = int(temporal_timestamp)
            time_stamp_float = time_stamp_int / 10**6
            return time_stamp_float,precision
        
    @staticmethod
    def trans_timestamp_to_general_format(temporal_timestamp):
        if type(temporal_timestamp) == float:
            data_obj = datetime.fromtimestamp(temporal_timestamp)
            return data_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            if len(temporal_timestamp) == 16:
                data_obj = datetime.fromtimestamp(float(temporal_timestamp)/10**6)
                return data_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    @staticmethod
    def calculate_time_interval(period):
        # 将时间戳转换为 datetime 对象
        if period.start_time=='' or period.end_time=='':
            return ''
        dt1 = datetime.fromtimestamp(float(period.start_time))
        dt2 = datetime.fromtimestamp(float(period.end_time))
        
        # 计算时间间隔
        delta = dt2 - dt1
        
        # 提取时分秒
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours += delta.days*24
        # 格式化输出
        time_interval = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return time_interval
    
    @staticmethod
    def get_vec_start_end_timestamp(file_name,mode='vec'):
        #读取从数据包中解析出的各种数据类型。
        # return file_name
        if mode == 'vec':
            return file_name.split('_')[0],file_name.split('_')[1].split('.')[0]
        if mode == 'traj':
            return file_name.split('_')[1].split('.')[0]

    def get_raw_data_package_timestamp(self,file_name):
        # extract timestamp from the file name of loc \ vision data
        name,suffix = file_name.split('.')
        items = name.split('_')
        try:
            date = items[2]
            time = items[3]
            time_str = date + ' ' + time
            time_format = "%Y-%m-%d %H-%M-%S"
            local_time = datetime.strptime(time_str, time_format)
            beijing_timestamp = local_time.timestamp()
            print("utc+8:", beijing_timestamp)
        except IndexError:
            return None

class CoordProcessor:
    @staticmethod
    def gcj02towgs84_point_level(lng, lat):
        """
        GCJ02(火星坐标系)转WGS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        def transformlat(lng, lat):
            ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
                0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
            ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                    math.sin(2.0 * lng * pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(lat * pi) + 40.0 *
                    math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
            ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
                    math.sin(lat * pi / 30.0)) * 2.0 / 3.0
            return ret
        def transformlng(lng, lat):
            ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
                0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
            ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                    math.sin(2.0 * lng * pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(lng * pi) + 40.0 *
                    math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
            ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
                    math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
            return ret
        def out_of_china(lng, lat):
            """
            判断是否在国内，不在国内不做偏移
            :param lng:
            :param lat:
            :return:d
            """
            if lng < 72.004 or lng > 137.8347:
                return True
            if lat < 0.8293 or lat > 55.8271:
                return True
            return False
        if out_of_china(lng, lat):
            return lng, lat
        dlat = transformlat(lng - 105.0, lat - 35.0)
        dlng = transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return lng * 2 - mglng, lat * 2 - mglat
    
    @staticmethod
    def trans_Zlevel_to_zero(z_json_path):
        with open(z_json_path, 'r') as geojson_file:
            geojson_data = json.load(geojson_file)
        if 'trajectory_'in z_json_path:
            #原始轨迹的geojson中最外层包了一个车辆id，此判断是取出其中的geojson
            geojson_data = geojson_data[list(geojson_data.keys())[0]]
        for feature in geojson_data['features']:
            # 获取要素的几何信息
            geometry = feature['geometry']

            # 检查是否包含 'coordinates' 字段
            if 'coordinates' in geometry:
                # 获取坐标列表
                coordinates = geometry['coordinates']
                # 遍历坐标列表并将高程设置为 0
                if geometry['type']=="Polygon":
                    for i in range(len(coordinates[0])):
                        coordinates[0][i][2] = 0
                elif geometry['type']=="LineString":
                    for i in range(len(coordinates)):
                        coordinates[i][2] = 0
                elif geometry['type']=="Point":
                    coordinates[2] = 0
    
        # 将修改后的数据保存回 GeoJSON 文件
        trans_path = z_json_path.replace('_feature','_feature_noH')

        with open(trans_path, 'w') as output_file:
                json.dump(geojson_data, output_file, indent=2)
        # _if_slice = -1
        # if 'noH' in trans_path and 'all_' not in trans_path:
        #     _if_slice = 1
        # else:
        #     _if_slice = 0
        # print(trans_path)
        # if _if_slice == 1:
        #     with open(trans_path, 'w') as output_file:
        #         json.dump(geojson_data, output_file, indent=2)
        # elif _if_slice ==0:
        #     with open(trans_path.replace('.geojson','noH.geojson'), 'w') as output_file:
        #         json.dump(geojson_data, output_file, indent=2)
        print("已将语义高程设置为 0 ，并保存到 _noH.geojson 文件中")

    @staticmethod
    def trans_gcj02towgs84(bias_json_path):
        with open(bias_json_path, 'r') as geojson_file:
            geojson_data = json.load(geojson_file)

        for feature in geojson_data['features']:
                # 获取要素的几何信息
                geometry = feature['geometry']
                # 检查是否包含 'coordinates' 字段
                if geometry != None:
                    if 'coordinates' in geometry :
                        # 获取坐标列表
                        coordinates = geometry['coordinates']
                        if geometry['type']=="Polygon":
                        # 遍历坐标列表并将高程设置为 0
                            for i in range(len(coordinates[0])):
                                coordinates[0][i][0],coordinates[0][i][1] = CoordProcessor.gcj02towgs84_point_level(coordinates[0][i][0],coordinates[0][i][1])
                        elif geometry['type']=="LineString":
                            for i in range(len(coordinates)):
                                coordinates[i][0],coordinates[i][1] = CoordProcessor.gcj02towgs84_point_level(coordinates[i][0],coordinates[i][1])
                        elif geometry['type']=="Point":
                            coordinates[0],coordinates[1] = CoordProcessor.gcj02towgs84_point_level(coordinates[0],coordinates[1])
        # with open(bias_json_path.replace('.geojson','_trans.geojson').replace('temp','out'), 'w') as output_file:
        #     json.dump(geojson_data, output_file, indent=2)
        # print("已将语义高程设置为 0，并保存到 _noH.geojson 文件中")j

        with open(bias_json_path.replace('.geojson','_trans.geojson'), 'w') as output_file:
            json.dump(geojson_data, output_file, indent=2)
        print("已转换坐标，并保存到 _trans.geojson 文件中")

    @staticmethod
    def trans_gcj02towgs84_replace(bias_json_path):
        with open(bias_json_path, 'r') as geojson_file:
            geojson_data = json.load(geojson_file)
        # 如果 geojson_data 最外层是一个单独的键值对，则将其返回至一面层的字典
        if isinstance(geojson_data, dict) and len(geojson_data) == 1:
            geojson_data = list(geojson_data.values())[0]
        for feature in geojson_data['features']:
                # 获取要素的几何信息
                geometry = feature['geometry']
                # 检查是否包含 'coordinates' 字段
                if geometry != None:
                    if 'coordinates' in geometry :
                        # 获取坐标列表
                        coordinates = geometry['coordinates']
                        if geometry['type']=="Polygon":
                        # 遍历坐标列表并将高程设置为 0
                            for i in range(len(coordinates[0])):
                                coordinates[0][i][0],coordinates[0][i][1] = CoordProcessor.gcj02towgs84_point_level(coordinates[0][i][0],coordinates[0][i][1])
                        elif geometry['type']=="LineString":
                            for i in range(len(coordinates)):
                                coordinates[i][0],coordinates[i][1] = CoordProcessor.gcj02towgs84_point_level(coordinates[i][0],coordinates[i][1])
                        elif geometry['type']=="Point":
                            coordinates[0],coordinates[1] = CoordProcessor.gcj02towgs84_point_level(coordinates[0],coordinates[1])
        # with open(bias_json_path.replace('.geojson','_trans.geojson').replace('temp','out'), 'w') as output_file:
        #     json.dump(geojson_data, output_file, indent=2)
        # print("已将语义高程设置为 0，并保存到 _noH.geojson 文件中")j

        with open(bias_json_path, 'w') as output_file:
            json.dump(geojson_data, output_file, indent=2)
        # input("已转换坐标，并保存到 _trans.geojson 文件中")
