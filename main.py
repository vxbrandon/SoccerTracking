import logging
import os
import warnings

import hydra
from omegaconf import OmegaConf, DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="configs", config_name="zap_vision")
def main(cfg: DictConfig):
    """
    Various configs (e.g. batch_size, model_type, etc.) are stored in yaml format in /project_dir/configs
    directory. For example, configurations related to yolov8 model is stored in configs/models/object_detection/yolov8.yaml.

    :param cfg: config file (yaml) loaded from configs.zap_vision.yaml
    :return:
    """
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
