import geopandas as gpd
import os

def geojson_to_sqlite(geojson_path):
    # 读取 GeoJSON 文件
    gdf = gpd.read_file(geojson_path)
    
    # 生成同名的 sqlite 文件路径
    sqlite_path = os.path.splitext(geojson_path)[0] + '.sqlite'
    
    # 将 GeoDataFrame 写入 SQLite 数据库
    # 使用 'sqlite' 引擎，表名设为 'data'
    gdf.to_file(sqlite_path, driver='SQLite', layer='data')
    
    print(f"Converted {geojson_path} to {sqlite_path}")
