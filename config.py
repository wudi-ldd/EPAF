# config.py
import torch

# Configuration parameters
config = {
    "model_type": 'vit_h',
    "sam_checkpoint": 'weights/sam_vit_h_4b8939.pth',
    "lora_rank": 32,  # Rank of LoRA
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "image_dir": 'datasets/images',
    "mask_dir": 'datasets/masks',
    "train_file": 'datasets/train.txt',
    "val_file": 'datasets/val.txt',
    "batch_size": 1,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "alpha": 0.5,  # Weight for contrastive center loss
    "checkpoint_path": 'logs/best_model_ce_lora.pth',
    "log_file": 'logs/best_model_ce_cocenter_lora.log',
    "warmup_epochs": 10,
    "min_lr_factor": 0.01,  # Minimum learning rate factor for cosine annealing
}
