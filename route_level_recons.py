
from src.reconstruction.egoview import EgoviewReconstruction,EgoviewReconsRouteScale
import yaml
from common_config import inference

if __name__ == "__main__":
    print("批量重建")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config.yaml", type=str, help='Path to config.yaml file')
    args = parser.parse_args()
    
    # tr = "/home/gyx/data/cqc/data4dev_use/slice_to_recons_02"
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config_paths = config['utdv2_settings']['target_dataset']

    for _dataset_index in range(len(config_paths)):
        reconstruction = EgoviewReconsRouteScale(config_path=args.config,Car_Type="CD701",_dataset_index=_dataset_index)
    # reconstruction = EgoviewReconstruction(config_path=args.config,Car_Type="CD701")
    # reconstruction.batch_ego_reconstruction(mask_mode="pkl")
        reconstruction.batch_recons_for_single_rao_dat()
    print("批量重建完成")