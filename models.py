# models.py
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from lora import LoRA_sam

class FeatureMapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapper, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

def load_sam_model(config):
    """Load the SAM model."""
    sam_model = sam_model_registry[config["model_type"]](checkpoint=config["sam_checkpoint"])
    sam_model.to(config["device"])
    return sam_model

def initialize_lora_sam_model(sam_model, config):
    """Initialize the LoRA_sam model."""
    lora_sam_model = LoRA_sam(sam_model, rank=config["lora_rank"])
    lora_sam_model.to(config["device"])
    return lora_sam_model
