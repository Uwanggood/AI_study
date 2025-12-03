from dataclasses import Field

import torch
from pydantic_settings import BaseSettings


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TutorialConfig(BaseSettings):
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    device: str = Field(default_factory=get_device)
    path_root: str = "./data"

    class Config:
        env_file = ".env"


config = TutorialConfig()