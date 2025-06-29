from pyproj import Transformer
import os
import json
import geojson
import time
from typing import List, Dict, Any
import numpy as np
# from protobuf_to_dict import protobuf_to_dict  # Removed as it is not accessed
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
import proto.DataCollection_pb2 as DataCollection_pb2
import proto.TileData_pb2 as tile_data_pb
import proto.Tencent_TileData_pb2 as tencent_tile_data_pb
from rtree import index
from shapely.geometry import shape
from tqdm import tqdm
import pickle
import yaml

VEC_FOLDER = "VecJsonData"
TRAJ_FOLDER = "TrajJsonData"
TARGET_FEATURES_TYPES = [
                        'HADLaneDivider_trans.geojson', \
                        'HADRoadDivider_trans.geojson', \
                        'LandMark_trans.geojson'
                        ]
cur_tmp = int(time.time() * 1000000)
class CarProtoConvert(object):
    """
        车端数据proto转geojson类
    """
    def __init__(self):
        self.precision1 = 100000000
        self.precision2 = 1
        self.max_x = 135.083333
        self.min_x = 73.55
        self.max_y = 53.55
        self.min_y = 3.85
        self.max_z = 8844450
        self.min_z = -154320

        self.detection_temp = []
        self.senmatic_list = []

    def get_rect_polygon(self, bounds):
        """
        根据元组(最小x，最小y，最大x，最大y)获取矩形框
        """
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        min_z = min_z / 1000
        max_z = max_z / 1000
        polygon_list = [[min_x, min_y, min_z], [max_x, min_y, max_z],
                        [max_x, max_y, max_z], [min_x, max_y, min_z],
                        [min_x, min_y, min_z]]

        return polygon_list

    def xcode_to_bit32(self, t):
        if '2' <= t <= '9':
            return ord(t) - ord('2')
        if 'A' <= t <= 'H':
            return ord(t) - ord('A') + 8
        if 'J' <= t <= 'N':
            return ord(t) - ord('J') + 16
        if 'P' <= t <= 'Z':
            return ord(t) - ord('P') + 21

        return 255

    def geo_range_check(self, x_point):
        if self.max_x < x_point.x or x_point.x < self.min_x:
            return False
        if self.max_y < x_point.y or x_point.y < self.min_y:
            return False
        if self.max_z < x_point.z or x_point.z < self.min_z:
            return False
        return True

    def convert_code_to_geo(self, m_code):
        if not m_code or 18 != len(m_code):
            return False
        x = ((((self.xcode_to_bit32(m_code[4]) & 0x01) << 2) | (
                self.xcode_to_bit32(m_code[5]) >> 3)) * 32 * 32 * 32 * 32 * 32 * 32) + \
            ((self.xcode_to_bit32(m_code[12])) * 32 * 32 * 32 * 32 * 32) + \
            ((self.xcode_to_bit32(m_code[13])) * 32 * 32 * 32 * 32) + \
            ((self.xcode_to_bit32(m_code[14])) * 32 * 32 * 32) + \
            ((self.xcode_to_bit32(m_code[15])) * 32 * 32) + \
            ((self.xcode_to_bit32(m_code[16])) * 32) + \
            (self.xcode_to_bit32(m_code[17]))
        y = (self.xcode_to_bit32(m_code[5]) & 0x07) * 32 * 32 * 32 * 32 * 32 * 32 + \
            (self.xcode_to_bit32(m_code[6])) * 32 * 32 * 32 * 32 * 32 + \
            (self.xcode_to_bit32(m_code[7])) * 32 * 32 * 32 * 32 + \
            (self.xcode_to_bit32(m_code[8])) * 32 * 32 * 32 + \
            (self.xcode_to_bit32(m_code[9])) * 32 * 32 + \
            (self.xcode_to_bit32(m_code[10])) * 32 + \
            (self.xcode_to_bit32(m_code[11]))
        z = (self.xcode_to_bit32(m_code[0])) * 32 * 32 * 32 * 32 * 32 + (
            self.xcode_to_bit32(m_code[1])) * 32 * 32 * 32 * 32 + (
                self.xcode_to_bit32(m_code[2])) * 32 * 32 * 32 + (
                self.xcode_to_bit32(m_code[3])) * 32 * 32 + (self.xcode_to_bit32(m_code[4])) * 32 + (
                self.xcode_to_bit32(m_code[5]))

        z >>= 6
        x = x / self.precision1 + self.min_x
        y = y / self.precision1 + self.min_y
        z = z / self.precision2 + self.min_z
        x_point = Point((x, y, z))
        if self.geo_range_check(x_point):
            return x_point
        else:
            return None

    def convert_feature(self, geometry, properties):
        feature = geojson.Feature(geometry=None, properties=properties)
        feature['geometry'] = json.loads(geojson.dumps(geometry))
        return feature

    def get_semantic_property(self, obj_data):

        property_list = {}

        # match obj_data.type:
        #     case DataCollection_pb2.ObjectType.ROAD_SURFACE_LINE:
        #         subtype = obj_data.lineType
        #     case DataCollection_pb2.ObjectType.ROAD_SURFACE_ARROW:
        #         subtype = obj_data.arrowType
        #     case DataCollection_pb2.ObjectType.ROAD_SURFACE_MARK:
        #         subtype = obj_data.markType
        #     case DataCollection_pb2.ObjectType.ROAD_SIGN:
        #         subtype = obj_data.signType
        #     case DataCollection_pb2.ObjectType.ROAD_TRAFFIC_LIGHT:
        #         subtype = obj_data.trafficLightType
        #     case DataCollection_pb2.ObjectType.ROAD_BOUNDARY:
        #         subtype = obj_data.boundaryType
        #     case DataCollection_pb2.ObjectType.ROAD_OVERHEAD:
        #         subtype = obj_data.overheadType
        #     case DataCollection_pb2.ObjectType.ROAD_POLE:
        #         subtype = obj_data.poleType
        #     case _:
        #         subtype = 1000
        if obj_data.type == DataCollection_pb2.ObjectType.ROAD_SURFACE_LINE:
            subtype = obj_data.lineType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_SURFACE_ARROW:
            subtype = obj_data.arrowType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_SURFACE_MARK:
            subtype = obj_data.markType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_SIGN:
            subtype = obj_data.signType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_TRAFFIC_LIGHT:
            subtype = obj_data.trafficLightType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_BOUNDARY:
            subtype = obj_data.boundaryType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_OVERHEAD:
            subtype = obj_data.overheadType
        elif obj_data.type == DataCollection_pb2.ObjectType.ROAD_POLE:
            subtype = obj_data.poleType
        else:
            subtype = 1000
        property_list['oid'] = str(obj_data.id)
        property_list['color'] = obj_data.color
        property_list['type'] = obj_data.type
        property_list['type_confidence'] = obj_data.type_confidence
        property_list['subtype_confidence'] = obj_data.subtype_confidence
        property_list['value'] = obj_data.value
        property_list['value_confidence'] = obj_data.value_confidence
        property_list['subtype'] = subtype
        property_list['longitudinal_typ'] = obj_data.lineLongitudinalType

        return property_list

    def convert_semantic(self, semantic_map_list, dir_name):
        for semantic_map in semantic_map_list.semantic_map_info_list:
            vec_fs = []
            obj_fs = []
            base_info = semantic_map.base_info
            start_time = base_info.start_time
            # end_time = start_time + base_info.work_time
            end_time_list = [obj_data.time_stamp.start_time + obj_data.time_stamp.work_time
                             for obj_data in semantic_map.objdata_list]
            if not end_time_list:
                continue
            end_time = max(end_time_list)
            if start_time < 0 or end_time < 0 or start_time > cur_tmp or end_time > cur_tmp:
                continue
            # print(semantic_map.base_info)

            for obj_data in semantic_map.objdata_list:
                point_list = []
                geometry = None

                if obj_data.HasField('outline'):
                    for m_point in obj_data.outline.point:
                        point = self.convert_code_to_geo(m_point.mcode)
                        if not point:
                            continue
                        point_list.append([point.x, point.y, point.z/1000])
                    geometry = Polygon(point_list)
                elif obj_data.HasField('line'):
                    for m_point in obj_data.line.point:
                        point = self.convert_code_to_geo(m_point.mcode)
                        point_list.append([point.x, point.y, point.z/1000])
                    geometry = LineString(point_list)
                elif obj_data.HasField('rect'):
                    lb = self.convert_code_to_geo(obj_data.rect.lb.mcode)
                    rt = self.convert_code_to_geo(obj_data.rect.rt.mcode)
                    if not lb and not rt:
                        continue
                    bounds = (lb.x, lb.y, lb.z,
                                rt.x, rt.y, rt.z)
                    point_list = self.get_rect_polygon(bounds)
                    geometry = Polygon(point_list)
                elif obj_data.HasField('center_point'):
                    mm_geometry = self.convert_code_to_geo(obj_data.center_point.mcode)
                    geometry = Point(mm_geometry.x, mm_geometry.y, mm_geometry.z/1000)
                # elif obj_data.HasField('solid'):
                #     top_point = self.convert_code_to_geo(obj_data.solid.point_top.mcode)
                #     bottom_point = self.convert_code_to_geo(obj_data.solid.point_bottom.mcode)
                #     point_list.append([top_point.x, top_point.y, top_point.z])
                #     point_list.append([bottom_point.x, bottom_point.y, bottom_point.z])
                #     geometry = MultiPoint(point_list)

                if not geometry:
                    continue

                property_list = self.get_semantic_property(obj_data)
                # if obj_data.source == DataCollection_pb2.Source.DETECTION:
                property_list['start_time'] = start_time
                property_list['end_time'] = end_time
                feature = self.convert_feature(geometry=geometry, properties=property_list)

                if obj_data.source == DataCollection_pb2.Source.DETECTION:
                    obj_fs.append(feature)
                elif obj_data.source == DataCollection_pb2.Source.SEMANTIC_RECOGNITION:
                    vec_fs.append(feature)

            # file_name = str(start_time) + '_' + str(end_time) + '.geojson'
            # obj_path = os.path.join(dir_name, 'ObjJsonData')
            # vec_path = os.path.join(dir_name, 'VecJsonData')

            # if len(obj_fs) > 0:
            #     print('obj: ' + str(start_time)  + ' ' + str(end_time))
            #     fc = geojson.FeatureCollection(obj_fs)
            #     file_path = os.path.join(obj_path, file_name)
            #     # print(file_path)
            #     with open(file_path, "w", encoding='utf-8') as file:
            #         file.write(json.dumps(fc))

            if len(obj_fs) > 0:
                self.detection_temp.append((start_time, end_time, obj_fs))

            if len(vec_fs) > 0:
                # print('vec: ' + str(start_time)  + ' ' + str(end_time))
                # fc = geojson.FeatureCollection(obj_fs + vec_fs)
                # file_path = os.path.join(vec_path, file_name)

                # obj_fs.clear()
                # with open(file_path, "w", encoding='utf-8') as file:
                #     file.write(json.dumps(fc))

                self.senmatic_list.append((start_time, end_time, vec_fs))

    def convert_semantic_data(self, path, dir_path):
        with open(path, "rb") as file:
            buf = file.read()
        collection_info = DataCollection_pb2.DataCollectInfo()
        collection_info.ParseFromString(buf)

        if collection_info.data_type == DataCollection_pb2.SEMANTIC_MAP:
            semantic_map_list = DataCollection_pb2.SemanticMapList()
            semantic_map_list.ParseFromString(collection_info.collect_data)
            self.convert_semantic(semantic_map_list, dir_path)

    def get_trajectory_property(self, location_info):
        property_list = {}

        property_list['pitch'] = location_info.pitch
        property_list['roll'] = location_info.roll
        property_list['timestamp'] = location_info.timestamp
        property_list['ve'] = location_info.neg_speed.east_velocity
        property_list['vn'] = location_info.neg_speed.north_velocity
        property_list['vg'] = location_info.neg_speed.ground_velocity
        property_list['azimuth'] = location_info.yaw

        return property_list

    def convert_trajectory(self, location_data, common_data, dir_name):
        fs = []
        for location_info in location_data.location_info_list:
            property_list = self.get_trajectory_property(location_info)

            mm_geometry = self.convert_code_to_geo(location_info.geo_point.mcode)
            geometry = Point(mm_geometry.x, mm_geometry.y, mm_geometry.z/1000)
            property_list['altitude'] = mm_geometry.z/1000

            feature = self.convert_feature(geometry=geometry, properties=property_list)
            fs.append(feature)

        base_info = location_data.base_info

        fc = geojson.FeatureCollection(fs)
        fc['car_id'] = common_data.id
        fc['model_id'] = common_data.model_id
        fc['start_time'] = base_info.start_time
        fc['duration_time'] = base_info.start_time + base_info.work_time
        traj = {}
        # traj['format_version'] = common_data.protocol_version
        traj[common_data.id] = fc

        file_name = 'trajectory_' + str(base_info.start_time) + '.geojson'
        file_path = os.path.join(dir_name, file_name)
        # print(file_path)

        with open(file_path, "w", encoding='utf-8') as file:
            file.write(json.dumps(traj))

    def convert_location_data(self, location_data, common_data, dir_name):

        location_list = []

        # base_info = location_data.base_info
        # print(base_info)

        for location_info in location_data.location_info_list:

            property_list = self.get_trajectory_property(location_info)

            mm_geometry = self.convert_code_to_geo(location_info.geo_point.mcode)
            geometry = Point(mm_geometry.x, mm_geometry.y, mm_geometry.z/1000)
            property_list['altitude'] = mm_geometry.z/1000

            # print(mm_geometry.x, mm_geometry.y)

            feature = self.convert_feature(geometry=geometry, properties=property_list)
            location_list.append((location_info.timestamp, feature, common_data))

        return location_list

    def read_all_location(self, trajectory_path):
        all_traj_list = []
        for top_path, _, files in os.walk(trajectory_path):
            file_num = len(files)
            sep_num = file_num // 100
            sep_list = [100 * sep for sep in range(0, sep_num + 1)] + [file_num]
            for n in range(len(sep_list) - 1):
                sub_traj_list = []
                sub_files = files[sep_list[n]:sep_list[n + 1]]
                for file in sub_files:
                    file_path = os.path.join(top_path, file)
                    with open(file_path, "rb") as file:
                        buf = file.read()
                    collection_info = DataCollection_pb2.DataCollectInfo()
                    collection_info.ParseFromString(buf)
                    if collection_info.data_type == DataCollection_pb2.LOCATION_INFO:
                        location_data = DataCollection_pb2.LocationData()
                        location_data.ParseFromString(collection_info.collect_data)

                        traj_list = self.convert_location_data(location_data, collection_info.common_data, file_path)

                        sub_traj_list = sub_traj_list + traj_list
                        traj_list.clear()
                all_traj_list += sub_traj_list
                sub_traj_list.clear()
        return all_traj_list

    def read_location(self, trajectory_file):
        with open(trajectory_file, "rb") as file:
            buf = file.read()
        collection_info = DataCollection_pb2.DataCollectInfo()
        collection_info.ParseFromString(buf)

        if collection_info.data_type == DataCollection_pb2.LOCATION_INFO:
            location_data = DataCollection_pb2.LocationData()
            location_data.ParseFromString(collection_info.collect_data)
            return self.convert_location_data(location_data, collection_info.common_data, trajectory_file)

    def export_traj_file(self, common_data, traj_list, traj_path, start, end):
        fc = geojson.FeatureCollection(traj_list)
        fc['car_id'] = common_data.id
        fc['model_id'] = common_data.model_id
        fc['start_time'] = start
        fc['duration_time'] = end - start

        traj = {}
        # traj['format_version'] = common_data.protocol_version
        traj[common_data.id] = fc

        file_name = 'trajectory_' + str(start) + '.geojson'
        file_path = os.path.join(traj_path, file_name)

        with open(file_path, "w", encoding='utf-8') as file:
            file.write(json.dumps(traj))

    def rename_semantic_file(self, start, end, new_start, root):
        file_name = str(start) + '_' + str(end) + '.geojson'
        file_path = os.path.join(root, file_name)
        new_name = str(new_start) + '_' + str(end) + '.geojson'
        new_file_path = os.path.join(root, new_name)

        print(file_path + ' -> ' + new_file_path)

        os.rename(file_path, new_file_path)

    def convert_data(self, path, dir_path):
        with open(path, "rb") as file:
            buf = file.read()
        collection_info = DataCollection_pb2.DataCollectInfo()
        collection_info.ParseFromString(buf)
        if collection_info.data_type == DataCollection_pb2.SEMANTIC_MAP:
            semantic_map_list = DataCollection_pb2.SemanticMapList()
            semantic_map_list.ParseFromString(collection_info.collect_data)

            self.convert_semantic(semantic_map_list, dir_path)
        elif collection_info.data_type == DataCollection_pb2.LOCATION_INFO:
            location_data = DataCollection_pb2.LocationData()
            location_data.ParseFromString(collection_info.collect_data)

            traj_path = os.path.join(dir_path, 'TrajJsonData')
            self.convert_trajectory(location_data, collection_info.common_data, traj_path)
    def calc_runtime(desc):
        def wrapper(func):
            def inner(*args, **kwargs):
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                print(f'{desc}运行耗时: {end - start} 秒')
                return res

            return inner

        return wrapper


    @calc_runtime(desc='[实时状态] -- 矢量解析工具 -- 原始数据转换至geojson耗时')
    def convert_all_data(self, dir_path, download_path):
        # root_path = os.path.dirname(dir_path)
        root_path = dir_path

        if not os.path.exists(root_path):
            os.mkdir(root_path)

        # obj_path = os.path.join(root_path, 'ObjJsonData')
        vec_path = os.path.join(root_path, 'VecJsonData')
        traj_path = os.path.join(root_path, 'TrajJsonData')

        # if not os.path.exists(obj_path):
        #     os.mkdir(obj_path)

        if not os.path.exists(vec_path):
            os.mkdir(vec_path)

        if not os.path.exists(traj_path):
            os.mkdir(traj_path)

        # for root, _, files in os.walk(download_path):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         convert_data(file_path, root_path)

        semantic_path = os.path.join(download_path, 'semantic')
        if not os.path.exists(semantic_path):
            print("error: no semantic data" + semantic_path)
            return

        trajectory_path = os.path.join(download_path, 'trajectory')
        if not os.path.exists(trajectory_path):
            print("error: no trajectory data")
            return

        for top_path, _, files in os.walk(semantic_path):
            for name in files:
                file_path = os.path.join(top_path, name)
                # print("序列化数据解析中..",name,end='\r')
                print(name,end='\r')
                self.convert_semantic_data(file_path, root_path)

        for _, end_time, obj_fs in self.detection_temp:
            for senmatic_start, senmatic_end, feature_list in self.senmatic_list:
                if end_time >= senmatic_start and end_time <= senmatic_end:
                    # print(str(start_time) + " in " + str(senmatic_start) + " - " + str(senmatic_end))
                    feature_list += obj_fs
                    break

        for senmatic_start, senmatic_end, feature_list in self.senmatic_list:
            print("[实时状态] -- 写入车端语义矢量:"+senmatic_start.__str__(),end="\r")
            fc = geojson.FeatureCollection(feature_list)
            file_name = str(senmatic_start) + '_' + str(senmatic_end) + '.geojson'
            # vec_path = os.path.join(dir_path, 'VecJsonData')
            file_path = os.path.join(vec_path, file_name)
            # print(file_path)
            with open(file_path, "w", encoding='utf-8') as handle:
                handle.write(json.dumps(fc))

        vec_timestamp_list = []
        for top_path, _, files in os.walk(vec_path):
            for file_name in files:
                pos =  file_name.rfind('.')
                name = file_name[:pos]
                items = name.split('_')
                vec_timestamp_list.append((int(items[0]), int(items[1])))

        vec_timestamp_list.sort(key=lambda x:x[0])

        all_locations = self.read_all_location(trajectory_path)
        all_locations.sort(key=lambda x:x[0])

        index = 0
        common_data = None
        fs = []
        traj_start = 0
        for key in vec_timestamp_list:
            start, end = key
            if start < 0 or end < 0 or start > cur_tmp or end > cur_tmp:
                continue
            for item in all_locations:
                timestamp = item[0]
                traj = item[1]
                if common_data is None:
                    common_data = item[2]

                if timestamp >= start and timestamp <= end:  # 表示轨迹点时间戳在语义时间戳范围内
                    if len(fs) == 0:
                        if timestamp > start:
                            # 表示第一个轨迹点时间戳大于第一个semantic start timestamp，修改semantic文件名
                            self.rename_semantic_file(start, end, timestamp, vec_path)
                            print(str(start) + ' -> ' + str(timestamp))
                            traj_start = timestamp
                        else:
                            traj_start = start
                    fs.append(traj)

            if len(fs) > 0:  # 语义时间戳范围内轨迹点数量大于0时输出对应轨迹文件
                self.export_traj_file(common_data, fs, traj_path, traj_start, end)
                print("写入车端轨迹矢量:",traj_start.__str__(),end = "\r")
            fs.clear()
            common_data = None
            index = index + 1

def process_record(root):
    # 解析轨迹与矢量数据
    all_vecs = sorted(os.listdir(root))
    target_vec_list = []
    for target_vec in all_vecs:
        data_path = os.path.join(root, target_vec)
        current_car_data_dir_list = sorted(os.listdir(data_path))
        def count_files_by_extension(path, extension):
            count = 0
            for root, _, files in os.walk(path):
                count += sum(1 for f in files if f.endswith(extension))
            return count

        for dir in current_car_data_dir_list:
            target_path = os.path.join(data_path, dir)
            target_vec_list.append(target_path)
            download_path = target_path
            
            # Count .geojson and .record files
            geojson_count = count_files_by_extension(target_path, '.geojson')
            record_count = count_files_by_extension(target_path, '.record')
            
            # Skip conversion if more .geojson files than .record files
            if geojson_count > record_count and record_count > 0:
                print(f"[实时状态] -- 跳过已处理目录: {target_path}")
                continue
                
            print(f"[实时状态] -- 矢量数据解析:  {target_path}")
            converter = CarProtoConvert()
            print(f"[实时状态] -- 矢量数据解析当前处理目录: : {target_path}")
            converter.convert_all_data(target_path, download_path)
    return target_vec_list[0]

# 用于构建空间索引，加速车端数据的遍历过程
def ST_build_index(vehicle_data_items, index_path='spatial_index'):
    print("[实时状态] -- 索引构建开始")
    idx_dict = {}
    
    num_runs = len(vehicle_data_items.items())
    for _name in range(num_runs):
        print(index_path+'_'+str(_name))
        index_name = index_path+'_'+str(_name)
        if os.path.exists(index_name+'.dat'):
            p = index.Property()
            p.storage = index.RT_Disk
            p.filename = index_name
            idx_dict[_name] = index.Index(index_name, properties=p)
    if not idx_dict:
        print(f"[实时状态] -- 保存当前索引到{index_path}")
        p = index.Property()
        p.storage = index.RT_Disk
        p.filename = index_path

        for _key, vehicle_data in vehicle_data_items.items(): 
            p = index.Property()
            p.storage = index.RT_Disk
            p.filename = index_path+'_'+str(_key)
            
            oid_list = []
            __count = 0
            idx = index.Index(p.filename,properties=p)
            for target_vec in tqdm(vehicle_data, desc="Building index"):
                feature = target_vec['feature']
                geometry = shape(feature['geometry'])
                if feature['properties']['oid'] in oid_list:
                    oid_list.append(int(feature['properties']['oid']))
                else:
                    oid_list.append(-1)
                __count += 1
                idx.insert(int(feature['properties']['oid']), geometry.bounds)
            idx_dict[_key] = idx
            # idx.close()
        return idx_dict
    else:
        print(f"[实时状态] -- 读取已存在的索引{index_path}")
        return idx_dict
class HD_Data:
    def __init__(self, data_dir: str):
        """
        Initialize HD_Data class with directory path containing GeoJSON files
        
        Args:
            data_dir (str): Path to directory containing GeoJSON files
        """
        self.data_dir = data_dir
        # self.geojson_data = []

    def load_geojson_files(self) -> None:
        """
        Load all GeoJSON files from the specified directory
        """
        try:
            # Get all .geojson files in directory
            geojson_files = glob.glob(os.path.join(self.data_dir, "*.geojson"))
            
            for file_path in geojson_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.geojson_data.append(data)
                    
        except Exception as e:
            print(f"Error loading GeoJSON files: {str(e)}")

    def get_all_data(self) -> List[Dict[str, Any]]:
        """
        Return all loaded GeoJSON data
        
        Returns:
            List[Dict[str, Any]]: List of loaded GeoJSON objects
        """
        return self.geojson_data

    def __len__(self) -> int:
        """
        Return number of loaded GeoJSON files
        
        Returns:
            int: Number of GeoJSON files loaded
        """
        return len(self.geojson_data)
    
class VehicleData(HD_Data):
    def __init__(self, config_folder: str, data_dir: str ='UseConfig'):
        """
        Initialize VehicleData class with directory path containing GeoJSON files
        
        Args:
            data_dir (str): Path to directory containing GeoJSON files
        """
        super().__init__(data_dir)
        # self.data_dir = 'MultiFolder'
        self.__load_vehicle_data(config_folder)

    def __load_vehicle_data(self,config_path) -> None:
        """
        Load all GeoJSON files from the specified directory
        """
        # self.vehicle_data = []
        self.target_vec_list = []
        try:
            with open(config_path, 'r') as yaml_file:
                config = yaml.safe_load(yaml_file)
            target_paths = config['vector_data']['target_path']

            for target_path in target_paths:
                self.target_vec_list.append(process_record(target_path))

        except Exception as e:
            print(f"Error loading GeoJSON files: {str(e)}")
        # print("load vehicle data")

    def get_vec_data(self):
        self.meta_data = {}


        vehicle_data_in_runs = {}
        run_id = 0
        for target_path in self.target_vec_list:
            self.meta_data[run_id] = target_path
            _vec_path = os.path.join(target_path,VEC_FOLDER)
            # _traj_path = os.path.join(target_path,TRAJ_FOLDER)
            # Read all geojson files in trajectory folder
            vec_files = glob.glob(os.path.join(_vec_path, "*.geojson"))
            vehicle_data = []
            # Process each trajectory file
            for vec_file in vec_files:
                # print(vec_file)
                with open(vec_file, 'r') as f:
                    vec_data = json.load(f)
                    
                    # For each feature in the trajectory data
                    for feature in vec_data['features']:
                        # TODO: Add spatial indexing logic here
                        # Could use rtree or other spatial indexing library
                        # Example:
                        # bbox = feature['geometry'].bounds
                        # spatial_index.insert(feature_id, bbox)
                        
                        # Store feature data
                        vehicle_data.append({
                            'type': 'vecs',
                            'file': os.path.basename(vec_file),
                            'feature': feature
                        })
            vehicle_data_in_runs[run_id] = vehicle_data
            run_id += 1
        self.vehicle_data = vehicle_data_in_runs

        self.vehicle_data_oid2feature = {}
        for run_id, vec_data in self.vehicle_data.items():
            self.vehicle_data_oid2feature[run_id] = {}
            for vec_id, vec in enumerate(vec_data):
                self.vehicle_data_oid2feature[run_id][vec['feature']['properties']['oid']] = vec['feature']
            # print(self.vehicle_data)

    def build_rtree(self):
        # self.oid_to_feature = {}
        self.ST_index = ST_build_index(self.vehicle_data)
        pass

    # def get_target_maplearning_data(self,maplearn_data:MapLearningData):
    #     current_maplearning_target_tile = maplearn_data.tile_name
    #     for item in self.tiles['features']:
    #         if item['properties']['name'] == current_maplearning_target_tile:
    #             self.target_tile_name = current_maplearning_target_tile
    #             self.target_tile = shape(item['geometry'])
    #             self.target_tile_bounds = self.target_tile.bounds
    #             break
    #     self.bounding_maplearning_data = maplearn_data
    
    def get_items_in_tile(self):
        """
        Get all vehicle items that intersect with the target tile
        Returns a dict of lists containing items from each run that intersect with the tile
        """
        tile_bounds = self.target_tile_bounds
        items_in_tile = {}
        
        # 遍历每个run的R-tree索引
        for run_id, RT_index in self.ST_index.items():
            # 初始化当前run的结果列表
            items_in_tile[run_id] = []
            
            # 使用generator获取相交的要素ID
            for vec_id in RT_index.intersection(tile_bounds):
                # 从原始数据中获取完整的要素信息
                vehicle_item = self.vehicle_data_oid2feature[run_id][str(vec_id)]
                items_in_tile[run_id].append(vehicle_item)
                print(run_id,vehicle_item['properties']['oid'],end='\r')
            # 打印每个run中找到的要素数量    
            print(f"Found {len(items_in_tile[run_id])} items in run {run_id}")
        self.items_in_tile = items_in_tile