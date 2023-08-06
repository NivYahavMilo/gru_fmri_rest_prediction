from enum import Enum

from pydantic import BaseModel
import torch

class TrainingMode(Enum):
    ROI = 'ROI'
    NETWORK = 'NETWORK'


class RoiTrainingParams(BaseModel):
    train_data: str
    test_data: str
    voxels: int
    roi: str
    k_class: int
    mode: str
    k_hidden: int
    k_layers: int
    batch_size: int
    num_epochs: int
    train_size: int
    device: torch.device



class NetworkTrainingParams(BaseModel):
    train_data: str
    test_data: str
    k_roi: int
    network: str
    k_class: int
    mode: str
    k_hidden: int
    k_layers: int
    batch_size: int
    num_epochs: int
    train_size: int
    device: torch.device

