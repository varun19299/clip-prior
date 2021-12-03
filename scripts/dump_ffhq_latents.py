from pathlib import Path
from models.stylegan import Generator

import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    plot_img = True
