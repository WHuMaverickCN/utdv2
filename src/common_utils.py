import json
import os
import math
from rtree import index
from shapely.geometry import shape
from tqdm import tqdm

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率

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
    