import logging
import os
import time

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    env = submitit.JobEnvironment()
    log.info(f"Process ID {os.getpid()} executing task with {env}")
    log.info("----- Hydra Config -----")
    log.info(OmegaConf.to_yaml(cfg))
    time.sleep(1)


if __name__ == "__main__":
    my_app()