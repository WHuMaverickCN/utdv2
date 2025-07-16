import cv2
import json
import geojson
import numpy as np
from pathlib import Path
import pandas as pd
from pyproj import Transformer
from src.common_utils import CoordProcessor
from scipy.spatial.transform import Rotation as sciR
np.set_printoptions(precision=6, suppress=True)
from math import pi, \
                radians, cos, sin, asin, sqrt
# from ..utils import print_run_time
from ..io import m_input

_lon = 106.4417403
_lat = 29.49942031

CHANGAN_EGO_TO_WORLD = {
    "rx":0.008,
    "ry":pi/2+0.003,
    "rz":-pi/2+0.001
}
CHANGAN_RAW_RPH = {
    "tx":1.77,
    "ty":0.07,
    "tz":1.34,
    "rx":-0.03037,
    "ry":0.028274+pi/2,
    "rz":-0.006632-pi/2
}

ARGOVERSE_RAW_QUAT = {
    "qw":0.502809,
    "qx":-0.499689,
    "qy":0.500147,
    "qz":-0.497340
}

ARGOVERSE_RAW_TRANS = {
    "tx":1.631216,
    "ty":-0.000779,
    "tz":1.432780
}

SAMPLE_POINTS_IN_PIXEL = {
    "p1_left_a":(1281, 1581),
    "p2_left_b":(1442, 1463),
    "p3_left_c":(1551, 1260),
    "p4_left_d":(1479, 1172),
    "p5_right_e":(1734, 1163),
    "p6_right_f":(2035, 1246),
    "p7_right_g":(2292, 1375),
    "p8_right_h":(2608, 1555)
    }

def from_rotation_vector_to_rotation_matrix(rvec):
    R = cv2.Rodrigues(rvec)[0]

def from_quanternion_to_euler(quanternion):
    rot = sciR.from_quat(quanternion)
    euler = rot.as_euler('xyz', degrees=True)
    print(euler)

def pose_to_extrinsic(pose_matrix):
    """
    将相机位姿矩阵转化为外参矩阵。

    参数：
    pose_matrix (numpy.ndarray): 3x4 的相机位姿矩阵 [R | t]

    返回：
    extrinsic_matrix (numpy.ndarray): 4x4 的外参矩阵
    """
    # 检查输入矩阵的形状
    if pose_matrix.shape != (3, 4):
        raise ValueError("位姿矩阵的形状应为 3x4")

    # 构建 4x4 外参矩阵
    extrinsic_matrix = np.eye(4)  # 创建一个单位矩阵
    extrinsic_matrix[:3, :4] = pose_matrix  # 将 3x4 位姿矩阵赋值给外参矩阵的前 3 行

    return extrinsic_matrix

def triangulation():
    pass

from_rotation_vector_to_rotation_matrix(np.array(
    [CHANGAN_RAW_RPH["rx"],\
    CHANGAN_RAW_RPH["ry"],\
    CHANGAN_RAW_RPH["rz"]])
    )

def extrinsic_to_pose(R, t):
    """
    将外参矩阵转换为位姿矩阵。

    参数：
    R (numpy.ndarray): 3x3 的旋转矩阵
    t (numpy.ndarray): 3x1 的平移向量

    返回：
    pose_matrix (numpy.ndarray): 4x4 的位姿矩阵
    """
    # 计算旋转矩阵的转置
    R_inv = R.T
    
    # 计算新的平移向量
    t_inv = -R_inv @ t

    # 构建 4x4 位姿矩阵
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R_inv
    pose_matrix[:3, 3] = t_inv.flatten()

    return pose_matrix

def camera_pose_to_extrinsic(_rotation_matrix:np.array, _translation_vector:np.array):
    """
    将相机位姿矩阵转换为外参矩阵。
    """

    # 计算旋转矩阵的逆（转置）
    rotation_matrix_inv = _rotation_matrix.T

    # 计算平移向量的逆：-R^T * t
    translation_vector_inv = -np.dot(rotation_matrix_inv, _translation_vector)

    extrinsic_matrix = np.hstack((rotation_matrix_inv, translation_vector_inv.reshape(3, 1)))
    return extrinsic_matrix

def pose_to_extrinsic(pose_matrix):
    """
    将位姿矩阵转换为外参矩阵。

    参数：
    pose_matrix (numpy.ndarray): 4x4 的位姿矩阵

    返回：
    R_extrinsic (numpy.ndarray): 3x3 的旋转矩阵
    t_extrinsic (numpy.ndarray): 3x1 的平移向量
    """
    # 提取位姿矩阵的旋转部分和平移部分
    R_pose = pose_matrix[:3, :3]
    t_pose = pose_matrix[:3, 3]

    # 计算旋转矩阵的转置（逆矩阵）
    R_extrinsic = R_pose.T

    # 计算外参平移向量
    t_extrinsic = -R_extrinsic @ t_pose

    return R_extrinsic, t_extrinsic

@staticmethod
def pixel_to_world_coords(u, v, camera_matrix, dist_coeffs, R, T, vehicle_height):
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
    # scale_factor = vehicle_height / (R[2, 0] * normalized_camera_coords[0] + 
    #                                 R[2, 1] * normalized_camera_coords[1] + 
    #                                 R[2, 2])
    scale_factor = vehicle_height / np.dot(R[2], normalized_camera_coords)

    # 乘以比例因子得到相机坐标系中的点
    camera_coords_scaled = normalized_camera_coords * scale_factor

    # 应用外参变换，将相机坐标系坐标转换到世界坐标系
    world_coords = np.dot(R, camera_coords_scaled) + T

    # 返回世界坐标
    return camera_coords_scaled,world_coords[:2]  # 通常假设z=0，返回x和y坐标

if __name__ == "__main__":
    # 通过四元数，构建相机位姿矩阵
    quanternion = [
        ARGOVERSE_RAW_QUAT["qx"],
        ARGOVERSE_RAW_QUAT["qy"],
        ARGOVERSE_RAW_QUAT["qz"],
        ARGOVERSE_RAW_QUAT["qw"],
        ]
    
    trans = [
        ARGOVERSE_RAW_TRANS["tx"],
        ARGOVERSE_RAW_TRANS["ty"],
        ARGOVERSE_RAW_TRANS["tz"]
    ]
    print("**********Argo 相机参数计算结果***********")
    rot = sciR.from_quat(quanternion)
    print(rot.as_euler('xyz', degrees=True))
    print(rot.as_euler('zxy', degrees=True))
    print(rot.as_euler('zyx', degrees=True))
    
    rot_mat = rot.as_matrix()
    rot_vec = rot.as_rotvec(degrees=True)
    print(f"根据四元数计算的旋转向量结果为：\n{rot_vec}")
    trans_vec = np.array(trans)
    extrinsic_matrix_rt = camera_pose_to_extrinsic(rot_mat,trans_vec)
    test_pose = extrinsic_matrix_rt @ (np.array([[0,0,0,1]]).T)

    inv_r = sciR.from_matrix(extrinsic_matrix_rt[:3,:3])
    print(f"根据位姿矩阵求逆的逆向旋转矩阵结果为：\n{inv_r.as_rotvec(degrees=True)}")
    print(test_pose)
    
    print("**********Changan 相机参数计算结果***********")

    # 通过长安提供的相机位姿参数构建旋转，声明的旋转顺序为 zyx
    rot_ca = sciR.from_euler('zyx',[
                                CHANGAN_RAW_RPH["rz"],\
                                CHANGAN_RAW_RPH["ry"],\
                                CHANGAN_RAW_RPH["rx"]
                                ])

    rot_vec_ca = rot_ca.as_rotvec(degrees=False)
    rot_mat_ca = rot_ca.as_matrix()
    print(f"根据四元数计算的旋转向量结果为：\n{rot_vec_ca}")
    print(rot_ca.as_euler('xyz', degrees=False))
    print(rot_ca.as_euler('zyx', degrees=False))
    print(rot_ca.as_euler('zxy', degrees=False))

    print(rot_ca.as_quat())
    trans_ca = [
        CHANGAN_RAW_RPH["tx"],
        CHANGAN_RAW_RPH["ty"],
        CHANGAN_RAW_RPH["tz"]
    ]
    trans_vec_ca = np.array(trans_ca)
    extrinsic_matrix_rt_ca = camera_pose_to_extrinsic(rot_mat_ca,trans_vec_ca)
    rot_mat = rot.as_matrix()

def trans_ego_to_world_coord(
        point_vehicle:np.ndarray,
        quanternion:list,
        geographical_coords:list,
        mode:str= "quat"
    )->np.ndarray:
    '''
    # 将车端x,y,z坐标转化为世界坐标
    # 输入一个点，输出一个点
    # 此处输入的四元数为长安汽车官方ins数据中解析得到，通常每一帧均需要读取一个四元数
    # 输入quanterion为[x,y,z,w]形式的列表

    如果 mode = "quat",则输入的quanternion为[x,y,z,w]形式的列表
    如果 mode = "euler",则读取旋转信息的为[x,y,z]形式的列表
    '''
    if mode == "quat":
        quat = quanternion[0]
        rot = sciR.from_quat(quat)
    elif mode == "euler":
        euler = quanternion[1]
        rot = sciR.from_euler('xyz',euler,degrees = True)
    else:
        return
    
    # print(rot.as_euler('xyz', degrees=True))
    # return 

    # 下面一套过程是根据四元数（quat）进行计算的，将自车坐标系下的点转换到世界坐标系下
    rot_matrix = rot.as_matrix()
    rot_matrix_inv = np.linalg.inv(rot_matrix)
    # rot_matrix_inv = rot_matrix
    point_world = np.dot(rot_matrix_inv, point_vehicle)
    utm_point_x,utm_point_y = from_raw_point_world_to_utm(point_world[0],point_world[1])
 
    # 即x表示东方向，y表示北方向
    ins_x,ins_y = from_wgs84_to_target_proj(geographical_coords[1],geographical_coords[0])

    # x方向为北，y方向为西
    # 该函数将自车坐标拽到x方向为北，y方向为西的的坐标系下
    return utm_point_x+ins_x,utm_point_y+ins_y

def from_raw_point_world_to_utm(raw_x,raw_y):
    new_x = -raw_y
    new_y = raw_x
    # 输入一个点，输出一个点
    # 该函数将自车坐标拽到x方向为东，y方向为北的的坐标系下
    return new_x,new_y

def from_wgs84_to_target_proj(
        lat,
        lon,
        source_proj = "epsg:4326",
        target_proj = "epsg:32648"):
    # 输入一个经纬度，输出投影坐标，默认为根据地理坐标系转为重庆地区(102°E-108°E)的CGCD2000-6度带投影坐标系 WGS 84 / UTM zone 48N
    # 该坐标为东、北坐标系，即Axes: Easting, Northing (E,N)
    # 即x表示东方向，y表示北方向
    # x通常为六位数，y通常为七位数，否则改坐标通常存在问题
    m_transformer = Transformer.from_crs(source_proj,target_proj)
    x,y = m_transformer.transform(lat,lon)
    # print(x,y)
    return x,y

def trans_instance_to_shape():
    # 输入一个EPSG42638的点集，输出一个矢量图层，该图层包含所有实例对应的矢量图形
    pass

# from_wgs84_to_target_proj(_lat,_lon)
# trans_ego_to_world_coord((0,0,0))
# @print_run_time("match_trajectory_to_insdata")
def ins_trans_util(ins_data_file:pd.DataFrame):
    """
    将ins数据文件转换为适合匹配的格式
    """
    # 读取ins数据文件
    ins_data_df = pd.read_csv(ins_data_file)
    
    # 提取需要的列
    # Extract the needed columns
    ins_data_df = ins_data_df[["sec_of_week", "utc", "latitude", "longitude"]]
    
    # # Create new fields with values from latitude and longitude
    # ins_data_df["new_latitude"] = ins_data_df["latitude"]
    # ins_data_df["new_longitude"] = ins_data_df["longitude"]

    def convert_row(row):
        new_lng, new_lat = CoordProcessor.gcj02towgs84_point_level(row['longitude'], row['latitude'])
        return pd.Series([new_lng, new_lat], index=['new_longitude', 'new_latitude'])
    ins_data_df[['new_longitude', 'new_latitude']] = ins_data_df.apply(convert_row, axis=1)
    # Swap the order of new_longitude and new_latitude columns
    cols = list(ins_data_df.columns)
    lng_idx = cols.index('new_longitude')
    lat_idx = cols.index('new_latitude')
    cols[lng_idx], cols[lat_idx] = cols[lat_idx], cols[lng_idx]
    ins_data_df = ins_data_df[cols]
    # 返回处理后的DataFrame
    return ins_data_df

def match_trajectory_to_insdata(trajectory:str,
                                ins_data_file:pd.DataFrame):
    print(trajectory)
    print(ins_data_file)
    _loc_path = str(Path(ins_data_file).absolute())
    loc_data_df = m_input.read_loc_data(_loc_path)

    with open(trajectory, 'r') as fp:
        _traj = geojson.load(fp)
    
    # 解析 B 数据中的点
    B_data = []

    # 如果traj.geojson文件中包含车辆编号，则取内层的数据进行重建
    if "features" not in _traj.keys():
        if len(_traj.keys())==1:
            for _ikey in  _traj.keys():
                _traj = _traj[_ikey]

    for feature in _traj["features"]:
        if feature["geometry"]["type"] == "Point":
            B_data.append(feature["geometry"]["coordinates"][:2])
            # print(feature["geometry"]["coordinates"])
            # B_data = feature["geometry"]["coordinates"]
    # 将 B 数据转换为 DataFrame
    B_data = pd.DataFrame(B_data, columns=['longitude', 'latitude'])
    A_data = loc_data_df[["sec_of_week","utc","new_latitude", "new_longitude"]]

    # 确定 A 数据长度和 B 数据长度
    A_len = len(A_data)
    B_len = len(B_data)

    points_per_segment = A_len / (B_len - 1)

    A_data_corrected = A_data.copy()

    def linear_interpolation(p1, p2, t):
        """ 计算 p1 和 p2 之间的插值点，t 为比例系数（0 <= t <= 1） """
        lat = (1 - t) * p1[1] + t * p2[1]
        lon = (1 - t) * p1[0] + t * p2[0]
        return lat, lon
    for i in range(A_len):
    # 找到 A 中的点位于 B 中的哪个线段
        segment_index = int(i // points_per_segment)
        
        if segment_index >= B_len - 1:
            # 确保 index 不超出 B 的范围
            segment_index = B_len - 2
        
        # 确定 A 点在 B 中对应的两点 B[segment_index] 和 B[segment_index + 1]
        B_start = B_data.loc[segment_index]
        B_end = B_data.loc[segment_index + 1]
        
        # 计算当前 A 点在该段上的插值比例 t
        t = (i % points_per_segment) / points_per_segment
        
        # 计算插值后的坐标
        new_lat, new_lon = linear_interpolation((B_start['longitude'], B_start['latitude']),
                                                (B_end['longitude'], B_end['latitude']),
                                                t)
        
        # 更新 A 中的点
        A_data_corrected.loc[i, 'latitude'] = new_lat
        A_data_corrected.loc[i, 'longitude'] = new_lon

    # 输出修正后的 A 数据
    return A_data_corrected
    print(A_data_corrected)
    print(loc_data_df)
    # 将ins_data匹配到traj_data，构建点到点的映射关系

def calculate_average_movement(df,uuid):
    # 计算每个点的移动向量
    df['delta_latitude'] = df['new_latitude'] - df['latitude']
    df['delta_longitude'] = df['new_longitude'] - df['longitude']
    
    # 计算每个点的移动角度（以度为单位）
    df['angle'] = np.arctan2(df['delta_longitude'], df['delta_latitude']) * (180 / np.pi)
    df['distance'] = df.apply(lambda row: haversine(row['longitude'], row['latitude'], row['new_longitude'], row['new_latitude']), axis=1)
    mean_angle = np.mean(df['angle'])
    
    # 计算所有点移动的平均值
    mean_movement = df['distance'].mean()
    # 将结果转换为JSON格式
    results = {
        'average_direction': mean_angle,
        'average_movement': mean_movement,
        'current_slice_uuid': uuid
    }
    
    return results

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两点之间的地表距离（单位：米）。
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000  # 地球平均半径，单位为米
    return c * r