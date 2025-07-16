import os
import json
import glob
import geojson
from shapely.geometry import shape
from typing import List, Dict, Any

from shapely.geometry import LineString,Polygon,Point
from pyproj import Transformer
from shapely.ops import transform
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import yaml

from src.common_utils import ST_build_index
import geopandas as gpd

TARGET_FEATURES_TYPES = [
                        'HADLaneDivider_trans.geojson', \
                        'HADRoadDivider_trans.geojson', \
                        'LandMark_trans.geojson'
                        ]


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
    
class MapLearningData(HD_Data):
    def __init__(self, data_dir: str, ground_truth_dir: str):
            """
            Initialize MapLearningData class with directories for data and ground truth
            
            Args:
                data_dir (str): Path to directory containing GeoJSON files
                ground_truth_dir (str): Path to directory containing ground truth files
            """
            super().__init__(data_dir)
            self.ground_truth_dir = ground_truth_dir
            self.__load_maplearn_data()
            self.__load_ground_truth()

    def __load_maplearn_data(self) -> None:
            """
            Load all GeoJSON files from the specified directory
            """
            self.maplearn_data = []
            try:
                # Get all .geojson files in main directory
                geojson_files = glob.glob(os.path.join(self.data_dir, "*.geojson"))
                
                # Get all .geojson files in semantic subdirectory
                semantic_dir = os.path.join(self.data_dir, "semantic")
                if os.path.exists(semantic_dir):
                    semantic_files = glob.glob(os.path.join(semantic_dir, "*.geojson"))
                    geojson_files.extend(semantic_files)
                
                for file_path in geojson_files:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.maplearn_data.append({
                            'type': os.path.basename(file_path).split('.')[0],
                            'data': data
                        })
            except Exception as e:
                print(f"Error loading GeoJSON files: {str(e)}")
            print("load map_learn_data")
    
    def __load_ground_truth(self) -> None:
            """
            Load ground truth data from specified directory
            """
            self.ground_truth_data = []
            _,tile_name = os.path.split(self.data_dir)

            # Get the full path of matching directory in ground truth folder
            matching_dir = os.path.join(self.ground_truth_dir, tile_name)
            self.tile_name = tile_name
            if os.path.exists(matching_dir):
                print(f"Found matching ground truth directory: {matching_dir}")
            else:
                print(f"No matching ground truth directory found for {tile_name}")

            try:
                # Read all geojson files in the matching directory
                # Only read files that match TARGET_FEATURES_TYPES
                for target_type in TARGET_FEATURES_TYPES:
                    print(target_type)
                    file_path = os.path.join(matching_dir, target_type)
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            # Extract feature type from filename
                            feature_type = os.path.basename(file_path).split('.')[0]
                            data = json.load(f)
                            # Store data with feature type as key
                            self.ground_truth_data.append({
                                'type': feature_type,
                                'data': data
                            })
                # print(self.ground_truth_data)
                print(f"Loaded {len(self.ground_truth_data)} ground truth feature types")
                        
            except Exception as e:
                print(f"Error loading ground truth files: {str(e)}")
            print("load ref ground truth data")
            # print(tile_name)

    def get_ground_truth(self) -> List[Dict[str, Any]]:
            """
            Return all loaded ground truth data
            
            Returns:
                List[Dict[str, Any]]: List of ground truth data
            """
            return self.ground_truth_data
    
    @staticmethod
    def ml_gt_compare(ml_feature,gt_features,_labeled_feature_num,label_result,pair,namely_ml_type):
                    min_distance = float('inf')
                    nearest_gt = None
                    
                    ml_feature['iou'] = None
                    
                    # Convert to list of coordinates if LineString
                    if ml_feature['geometry']['type'] == 'LineString':
                        ml_coords = ml_feature['geometry']['coordinates']
                        # ml_coords = [ml_coords[0]]  # Use first point for comparison
                        ml_line = LineString(ml_coords)
                        flag = 'line'
                    elif ml_feature['geometry']['type'] == 'Polygon':
                        if namely_ml_type == 'Line':
                            # 此处的设计是针对ml中Line类型的要素，但是内部的要素是Polygon类型的
                            # flag表征当成line类型，进行精度验证
                            flag = 'line'
                            ml_coords = ml_feature['geometry']['coordinates'][0]  # Get the outer ring coordinates
                            # ml_polygon = Polygon(ml_coords)
                            ml_line = LineString(ml_coords)
                        else:
                            # 此处对应的是正常情况
                            flag = 'polygon'
                            ml_coords = ml_feature['geometry']['coordinates'][0]  # Get the outer ring coordinates
                            ml_polygon = Polygon(ml_coords)
                            
                    # Find nearest ground truth feature
                    for gt_feature in gt_features:
                        gt_coords = gt_feature['geometry']['coordinates']
                        # Convert coordinates to Shapely objects
                        if gt_feature['geometry']['type'] == 'LineString':
                            gt_line = LineString(gt_coords)
                            gt_polygon = Polygon()
                        elif gt_feature['geometry']['type'] == 'Polygon':
                            gt_coords = gt_coords[0]  # Get the outer ring coordinates
                            gt_polygon = Polygon(gt_coords)
                        # Project coordinates from geographic to UTM (metric) for accurate distance calculation
                        
                        # Create transformer from WGS84 to UTM zone (重庆 area)
                        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
                        
                        if flag == 'line':
                            # Transform both lines to UTM coordinates
                            gt_line_utm = transform(transformer.transform, gt_line)
                            ml_line_utm = transform(transformer.transform, ml_line)
                            def split_line_into_segments(line: LineString) -> List[LineString]:
                                """
                                Split a line into 3 equal segments
                                Args:
                                    line: LineString object to split
                                Returns:
                                    List of 3 LineString segments
                                """
                                line_length = line.length
                                segment_length = line_length / 3
                                segments = []
                                # Get line coordinates
                                coords = list(line.coords)
                                # If too few points (less than 6), interpolate more points
                                if len(coords) < 6:
                                    # Calculate total length and desired point spacing
                                    total_length = line.length
                                    point_spacing = total_length / 5  # Get 6 points
                                    # Generate evenly spaced points
                                    coords = [line.interpolate(i * point_spacing) for i in range(6)]
                                
                                # Split into 3 segments
                                for i in range(3):
                                    start_dist = i * segment_length
                                    end_dist = (i + 1) * segment_length
                                    segment = LineString([line.interpolate(d) for d in [start_dist, end_dist]])
                                    segments.append(segment)
                                return segments
                            # Calculate distance between transformed lines
                            if namely_ml_type == 'Line'and ml_feature['geometry']['type'] == 'Polygon':
                                distance = ml_line_utm.distance(gt_line_utm)
                            else:
                            # Split line and calculate distances from centroids
                                segments = split_line_into_segments(ml_line_utm)
                                distances = [segment.centroid.distance(gt_line_utm) for segment in segments]
                                distance = sum(distances) / len(distances)
                            
                            # # Calculate distance between transformed lines
                            # distance = ml_line_utm.distance(gt_line_utm)

                            # Calculate distance between features
                            # distance = ((ml_coords[0][0] - gt_coords[0][0])**2 + 
                            #     (ml_coords[0][1] - gt_coords[0][1])**2)**0.5
                            
                            if distance < min_distance:
                                min_distance = distance
                                nearest_gt = gt_feature
                        elif flag == 'polygon':
                            if not isinstance(gt_polygon, Polygon) or gt_polygon.is_empty:
                                continue
                            # Transform polygons to UTM coordinates
                            ml_polygon_utm = transform(transformer.transform, ml_polygon)
                            gt_polygon_utm = transform(transformer.transform, gt_polygon)
                            
                            # Check if gt_polygon_utm is a valid Polygon and not empty
                            # if not isinstance(gt_polygon_utm, Polygon) or gt_polygon_utm.is_empty:
                            #     continue

                            # Calculate centroid distance
                            ml_centroid = ml_polygon_utm.centroid
                            gt_centroid = gt_polygon_utm.centroid
                            # Function to calculate projected distance between centroids
                            def calculate_projected_distance(ml_centroid, gt_centroid, gt_polygon_utm):
                                # Get the oriented minimum bounding rectangle of gt polygon
                                gt_bounds = gt_polygon_utm.minimum_rotated_rectangle
                                gt_coords = list(gt_bounds.exterior.coords)
                                
                                # Calculate the primary direction vector
                                # Using the longer edge of the bounding rectangle
                                dx1 = gt_coords[1][0] - gt_coords[0][0]
                                dy1 = gt_coords[1][1] - gt_coords[0][1]
                                dx2 = gt_coords[2][0] - gt_coords[1][0]
                                dy2 = gt_coords[2][1] - gt_coords[1][1]
                                
                                # Choose the longer edge as primary direction
                                if (dx1*dx1 + dy1*dy1) > (dx2*dx2 + dy2*dy2):
                                    direction_x, direction_y = dx1, dy1
                                else:
                                    direction_x, direction_y = dx2, dy2
                                    
                                # Normalize direction vector
                                length = (direction_x*direction_x + direction_y*direction_y)**0.5
                                direction_x /= length
                                direction_y /= length
                                
                                # Calculate perpendicular direction (-dy, dx)
                                perp_direction_x = -direction_y
                                perp_direction_y = direction_x
                                
                                # Project centroid difference onto perpendicular direction
                                diff_x = gt_centroid.x - ml_centroid.x 
                                diff_y = gt_centroid.y - ml_centroid.y
                                projected_distance = abs(diff_x*perp_direction_x + diff_y*perp_direction_y)
                                
                                return projected_distance

                            # Calculate centroid distance
                            centroid_distance = ml_centroid.distance(gt_centroid)

                            # projected_distance = calculate_projected_distance(ml_centroid, gt_centroid, gt_polygon_utm)
                            distance = centroid_distance

                            # Calculate IOU
                            intersection_area = ml_polygon_utm.intersection(gt_polygon_utm).area
                            union_area = ml_polygon_utm.union(gt_polygon_utm).area
                            iou = intersection_area / union_area if union_area > 0 else 0

                            # Use centroid distance as primary metric, but store IOU as well
                            
                            # ml_feature['iou'] = iou  # Store IOU for reference
                        if distance < min_distance:
                            min_distance = distance
                            nearest_gt = gt_feature

                            # 用于计算投影距离的临时参考要素
                            __proj_ml_centroid = ml_centroid
                            __proj_gt_polygon_utm = gt_polygon_utm
                            __proj_gt_centroid = gt_centroid

                            # print(ml_feature['properties']['id'],min_distance,nearest_gt['properties']['id'],end="\r")
                            ml_feature['iou'] = iou  # Store IOU for reference
                    if min_distance < 5:
                        if flag == 'polygon' and min_distance < 3.5:
                            projected_distance = calculate_projected_distance(__proj_ml_centroid, __proj_gt_centroid, __proj_gt_polygon_utm)
                            min_distance = projected_distance
                            label_result[pair['maplearn']['type']].update({ml_feature['properties']['id']:\
                                {
                                    'distance':min_distance,\
                                    'matched_gt_id':nearest_gt['properties']['id'],\
                                    'iou':ml_feature['iou'],\
                                    'feature_type':flag
                                }
                            })
                            _labeled_feature_num += 1
                        elif flag == 'line':
                            label_result[pair['maplearn']['type']].update({ml_feature['properties']['id']:\
                                {
                                    'distance':min_distance,\
                                    'matched_gt_id':nearest_gt['properties']['id'],\
                                    'feature_type':flag
                                }
                            })
                            _labeled_feature_num += 1
                        # print(min_distance,end="\r")    
                    # Add reference to nearest ground truth feature
                    ml_feature['nearest_gt'] = nearest_gt
                    ml_feature['distance_to_gt'] = min_distance
    def label(self,multi_thread=True):
        """
        Print maplearn_data and ground_truth_data
        """
        # print("Map Learn Data:")
        # for idx, data in enumerate(self.maplearn_data):
        #     print(f"Item {idx}:")
        #     print(data)
        #     print("-" * 50)
            
        # print("\nGround Truth Data:")
        # for idx, data in enumerate(self.ground_truth_data):
        #     print(f"Item {idx}:")
        #     print(data)
        #     print("-" * 50)

        # @staticmethod
        def associate_data() -> List[Dict[str, Any]]:
            """
            Associate maplearn_data with ground_truth_data based on feature types
            """
            associations = {
                'Line': 'HADLaneDivider_trans',
                'Boundary': 'HADRoadDivider_trans',
                'Arrow': 'LandMark_trans'
            }
            
            self.associated_data = []
            
            for ml_item in self.maplearn_data:
                ml_type = ml_item['type']
                if ml_type in associations:
                    # Find matching ground truth data
                    gt_type = associations[ml_type]
                    gt_item = next((item for item in self.ground_truth_data 
                                  if item['type'] == gt_type), None)
                    
                    if gt_item:
                        self.associated_data.append({
                            'maplearn': ml_item,
                            'ground_truth': gt_item
                        })
            
            print(f"Associated {len(self.associated_data)} feature pairs")
            return self.associated_data
        
        label_result = {}

        rel = associate_data()
        #此处的rel是一个list，每个元素是一个dict，包含了maplearn和ground_truth的数据
        # print(rel)
        for pair in rel:
            print('*******************Labeling-',pair['maplearn']['type'],'************')
            # 捉对进行匹配
            ml_features = pair['maplearn']['data']['features']
            # 需要注意的是，当 ml_features 的type为 Line 时，实际上内部的要素也会包含面状的 "Line"
            gt_features = pair['ground_truth']['data']['features']
            
            # 对于 Line 类型的 ml 进行特殊处理
            namely_ml_type = pair['maplearn']['type']

            label_result[pair['maplearn']['type']] = {}

            _labeled_feature_num = 0
            # For each maplearn feature, find nearest ground truth feature
            # Create a partial function with fixed arguments
           
            if multi_thread == False:
            # 1 单线程模式
                for ml_feature in tqdm(ml_features, 
                            desc=f"Matching {pair['maplearn']['type']}", 
                            total=len(ml_features),
                            position=0, 
                            leave=False,
                            dynamic_ncols=True):
                    MapLearningData.ml_gt_compare(ml_feature, 
                                    gt_features=gt_features,
                                    _labeled_feature_num=_labeled_feature_num,
                                    label_result=label_result,
                                    pair=pair,
                                    namely_ml_type=namely_ml_type)
            else:            
            # 2 多线程模式
                compare_func = partial(MapLearningData.ml_gt_compare, 
                                    gt_features=gt_features,
                                    _labeled_feature_num=_labeled_feature_num,
                                    label_result=label_result,
                                    pair=pair,
                                    namely_ml_type=namely_ml_type)

                # Use ThreadPoolExecutor to process features in parallel
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(tqdm(executor.map(compare_func, ml_features), 
                            desc=f"Matching {pair['maplearn']['type']}", 
                            total=len(ml_features),
                            position=0, 
                            leave=False,
                            dynamic_ncols=True))
            print("labeled_feature_rate:",_labeled_feature_num / len(ml_features),label_result)
            # print("labeled_feature_rate:",_labeled_feature_num )
        # Save label_result to JSON file
        @staticmethod
        def __extract_last_three_dirs(path: str) -> str:
            """
            Extract the last three directory levels from a path and join them with underscores
            
            Args:
                path (str): Input path string
            
            Returns:
                str: Last three directory levels joined with underscores
            """
            # Split path into components
            parts = path.rstrip('/').split('/')
            # Get last 3 parts and join with underscore
            return '_'.join(parts[-3:])
        suffix = __extract_last_three_dirs(self.data_dir)
        # Convert nested dict to flat dataframe
        rows = []
        for feature_type, features in label_result.items():
            for feature_id, data in features.items():
                row = {
                    'feature_type': feature_type,
                    'feature_id': feature_id,
                    **data  # Unpack all data fields
                }
                rows.append(row)
        
        # Create dataframe and save as CSV
        # Get current timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        df = pd.DataFrame(rows)
        df.to_csv(f'label_result_{suffix}_{timestamp}.csv', index=False)
        
        # Also save original json
        with open(f'label_result_{suffix}_{timestamp}.json', 'w') as f:
            json.dump(label_result, f, indent=4)
        return label_result

    def search_vehicle(self):
        """
        Search for vehicle features in maplearn_data
        """
        vehicle_features = []
        for item in self.maplearn_data:
            if item['type'] == 'Vehicle':
                vehicle_features.append(item)
        return vehicle_features
    
    def init_transform(self, target_epsg: str="EPSG:32648") -> None:
        transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
        self.transformer = transformer

    def search_overlap_lane_pointset_in_candidate(self, candidate_paths,namely_ml_type = 'Polygon'):
        """
        Given a list of candidate paths, read all geojson files under each path,
        print their file paths, and return a list of all found geojson files.
        """

        tf = self.transformer
        all_geojson_files = []
        for candidate_path in candidate_paths:
            geojson_files = glob.glob(os.path.join(candidate_path, "*.geojson"))
            for file_path in geojson_files:
                if os.path.basename(file_path) == "merged_result.geojson":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(f"{file_path}: {len(data.get('features', []))} features")
            all_geojson_files.extend(geojson_files)
        
        # 遍历 maplearn_data，打印每个成员的类型和要素个数
        for item in self.maplearn_data:
            item_type = item.get('type', 'Unknown')
            features = item.get('data', {}).get('features', [])
            if item_type in ['HADRoadDivider', 'HADLaneDivider']:
                print(f"Type: {item_type}, Feature count: {len(features)}")
                # 下一步处理逻辑可以在这里添加
                continue

        # 1. 读取所有候选geojson文件的点集要素
        print("Reading candidate point sets from geojson files...")
        for file_path in all_geojson_files:
            print(f"Reading candidate points from: {file_path}")
        candidate_points = []
        for file_path in all_geojson_files:
            try:
                gdf = gpd.read_file(file_path)
                # 只处理Point类型要素
                points_gdf = gdf[gdf.geometry.type == 'Point']
                candidate_points.append(points_gdf)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # 2. 判断maplearn_data成员是否和候选点距离小于1
        close_features = []
        # 遍历 maplearn_data，查找距离候选点小于1的 Point 要素，并显示进度条
        for item in self.maplearn_data:
            item_type = item.get('type', 'Unknown')
            if item_type not in ['HADRoadDivider', 'HADLaneDivider']:
                continue
            features = item.get('data', {}).get('features', [])
            for ml_feature in tqdm(features, desc=f"Checking {item_type} features", total=len(features), leave=False, dynamic_ncols=True):
                geom_type = ml_feature.get('geometry', {}).get('type')
                coords = ml_feature.get('geometry', {}).get('coordinates')

                if ml_feature['geometry']['type'] == 'LineString':
                    ml_coords = ml_feature['geometry']['coordinates']
                    # ml_coords = [ml_coords[0]]  # Use first point for comparison
                    ml_line = LineString(ml_coords)
                    flag = 'line'
                elif ml_feature['geometry']['type'] == 'Polygon':
                    if namely_ml_type == 'Line':
                        # 此处的设计是针对ml中Line类型的要素，但是内部的要素是Polygon类型的
                        # flag表征当成line类型，进行精度验证
                        flag = 'line'
                        ml_coords = ml_feature['geometry']['coordinates'][0]  # Get the outer ring coordinates
                        # ml_polygon = Polygon(ml_coords)
                        ml_line = LineString(ml_coords)
                    else:
                        # 此处对应的是正常情况
                        flag = 'polygon'
                        ml_coords = ml_feature['geometry']['coordinates'][0]  # Get the outer ring coordinates
                        ml_polygon = Polygon(ml_coords)

                if flag == 'line':
                    ml_line_utm = transform(tf.transform, ml_line)
                elif flag == 'polygon':
                    ml_polygon_utm = transform(tf.transform, ml_polygon)
                    # ml_centroid = ml_polygon_utm.centroid


                if geom_type == 'Point' or geom_type == 'LineString':
                    # 计算 ml_line_utm 到 candidate_gdf 每一个点要素的距离
                    _index_count = 0
                    for candidate_gdf in candidate_points:
                        # 只处理 Point 类型
                        candidate_pts = candidate_gdf[candidate_gdf.geometry.type == 'Point']
                        # 计算每个点到 ml_line_utm 的距离
                        distances = candidate_pts.geometry.apply(lambda pt: ml_line_utm.distance(pt))
                        # 找出距离小于1的点
                        close_pts = candidate_pts[distances < 1]
                        # 将最近点以gdf友好的方式存储（只存储点要素，方便后续gdf读写）
                        if not close_pts.empty:
                            feature_id = ml_feature.get('properties', {}).get('id', 'unknown')
                            candidate_file = os.path.basename(file_path)
                            save_dir = os.path.join(self.data_dir, "close_candidate_points")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{feature_id}_candidate_run_{str(_index_count)}_close_points.sqlite")
                            # 保存为SQLite文件，gdf可直接读取
                            close_pts.to_file(save_path, driver="SQLite", layer="close_points", index=False)
                        if not close_pts.empty:
                            close_features.append(ml_feature)
                            print(f"Current close_features count: {len(close_pts)}")
                            # break  # 已找到一个距离小于1的点就跳出
                        _index_count += 1
        print(f"Found {len(close_features)} maplearn features within distance < 1 to candidate points.")
                # print(f"Type: {item_type}, Feature count: {len(features)}")

        return all_geojson_files