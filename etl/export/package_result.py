import yaml
import os
import geopandas as gpd
import pandas as pd

from geojson2sqlite import geojson_to_sqlite
def main():
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取所有路径
    target_datasets = config.get('utdv2_settings', {}).get('target_dataset', [])
    for path in target_datasets:
        recons_result_dir = os.path.join(path, 'recons_result')
        if os.path.isdir(recons_result_dir):
            geojson_files = [f for f in os.listdir(recons_result_dir) if f.endswith('.geojson')]
            if len(geojson_files) >= 1:
                print(f"{recons_result_dir} contains {len(geojson_files)} .geojson file(s).")
                gdf_list = []
                for geojson_file in geojson_files:
                    geojson_path = os.path.join(recons_result_dir, geojson_file)
                    gdf = gpd.read_file(geojson_path)
                    gdf['source_file'] = geojson_file  # 可选：记录来源文件
                    gdf_list.append(gdf)
                if gdf_list:
                    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
                    print(type(merged_gdf), merged_gdf)
                    base_dir = os.path.dirname(recons_result_dir)
                    geojson_path = os.path.join(base_dir, 'merged_result.geojson')
                    merged_gdf.to_file(geojson_path, driver='GeoJSON', index=False)
                    geojson_to_sqlite(geojson_path)  # 转换为 SQLite 数据库
                    print(f"Merged GeoDataFrame saved to {geojson_path}")
            else:
                print(f"{recons_result_dir} does not contain any .geojson files.")
        else:
            print(f"{recons_result_dir} does not exist.")

if __name__ == '__main__':
    main()