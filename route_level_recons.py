
from src.reconstruction.egoview import EgoviewReconstruction,EgoviewReconsRouteScale

from common_config import inference

if __name__ == "__main__":
    print("批量重建")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config.yaml", type=str, help='Path to config.yaml file')
    args = parser.parse_args()
    
    # tr = "/home/gyx/data/cqc/data4dev_use/slice_to_recons_02"

    reconstruction = EgoviewReconsRouteScale(config_path=args.config,Car_Type="CD701")
    # reconstruction = EgoviewReconstruction(config_path=args.config,Car_Type="CD701")
    # reconstruction.batch_ego_reconstruction(mask_mode="pkl")
    print("批量重建完成")