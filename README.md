---

# EPAF: An Efficient Pore Annotation Framework for Tight Sandstone Images with the Segment Anything Model

## Overview

EPAF (Efficient Pore Annotation Framework) is an advanced automatic image annotation framework. It streamlines the annotation process for microscopy images, enhancing efficiency and accuracy in research workflows. By leveraging powerful models like SAM (Segment Anything Model) and integrating techniques like LoRA (Low-Rank Adaptation), EPAF provides a robust framework for researchers and practitioners in image segmentation and analysis.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Automatic Annotation](#automatic-annotation)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)
- [Semantic Segmentation Models](#semantic-segmentation-models)
- [Datasets](#datasets)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/EPAF.git
cd EPAF
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: Ensure you have PyTorch installed with CUDA support if you plan to train on a GPU.*

## Dataset Preparation

Organize your dataset in the following structure:

```
datasets/
│
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── masks/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── json/  # Optional for metadata or additional annotations
│   ├── image1.json
│   └── ...
│
├── train.txt  # List of image names (without extensions) used for training
└── val.txt    # List of image names (without extensions) used for validation
```

- **images/**: Input images in `.png` format. Images will be resized to 1024x1024 during data loading.
- **masks/**: Ground truth masks in `.png` format, corresponding to the images.
- **json/** (optional): Contains metadata or additional annotations in `.json` format.
- **train.txt** and **val.txt**: Text files listing the image names (without extensions) for training and validation.

Example content for `train.txt` or `val.txt`:

```
image1
image2
image3
...
```

## Training

### 1. Download the Pretrained SAM Model

- Visit the [SAM Model Repository](https://github.com/facebookresearch/segment-anything) to download the appropriate pretrained weights.
- Place the downloaded weights in the `weights/` directory.

### 2. Configure the Training Parameters

Modify the `config.py` file to adjust training parameters and paths according to your setup.

```python
# config.py
import torch

# Configuration parameters
config = {
    "model_type": 'vit_h',
    "sam_checkpoint": 'weights/sam_vit_h_4b8939.pth',
    "lora_rank": 16,  # Rank of LoRA
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
    "checkpoint_path": 'logs/best_model.pth',
    "log_file": 'logs/training.log',
    "warmup_epochs": 10,
    "min_lr_factor": 0.01,  # Minimum learning rate factor for cosine annealing
}
```

- **model_type**: Type of the SAM model to use (`'vit_h'`, `'vit_l'`, etc.).
- **sam_checkpoint**: Path to the pretrained SAM weights.
- **lora_rank**: Rank of the LoRA adaptation.
- **device**: Device to use for training (`'cuda'` or `'cpu'`).
- **image_dir**: Directory containing the training images.
- **mask_dir**: Directory containing the ground truth masks.
- **train_file** and **val_file**: Paths to the train and validation split files.
- **batch_size**: Batch size for training.
- **num_epochs**: Number of training epochs.
- **learning_rate**: Learning rate for the optimizer.
- **weight_decay**: Weight decay (L2 regularization) factor.
- **alpha**: Weight for the contrastive center loss component.
- **checkpoint_path**: Path to save the best model checkpoint.
- **log_file**: Path to save the training log.
- **warmup_epochs**: Number of epochs for the learning rate warmup phase.
- **min_lr_factor**: Minimum learning rate factor for cosine annealing.

### 3. Start Training

Run the training script:

```bash
python train.py
```

- The training process will log progress to the console and save detailed logs to `logs/training.log`.
- Model checkpoints and weights will be saved in the `logs/` directory.

### Training Details

- **Model Components**: During training, the SAM model's parameters are frozen. Only the LoRA layers and custom convolution layers are trained for feature extraction.
- **Loss Functions**: The training utilizes Cross-Entropy Loss combined with a Contrastive Center Loss for enhanced performance.
- **Learning Rate Scheduler**: Implements a warmup phase followed by cosine annealing for learning rate adjustment.

## Automatic Annotation

Once the model is trained, you can use it for automatic data annotation of new images.

### 1. Prepare New Images

Place the new images you wish to annotate in a directory, e.g., `new_images/`.

### 2. Configure the Annotation Parameters

Modify the `EPAF.py` script to set the paths and parameters for annotation.

```python
# EPAF.py

# Configuration parameters
config = {
    "sam_checkpoint": 'weights/sam_vit_h_4b8939.pth',
    "lora_sam_checkpoint": 'logs/best_lora_cocenter_rank16.safetensors',
    "model_type": 'vit_h',
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "image_folder": 'new_images',  # Directory containing new images for annotation
    "train_file": 'train.txt',     # Not used in annotation; can be ignored or set to None
    "predict_dir": 'predict',      # Directory to save prediction results
}
```

- **sam_checkpoint**: Path to the pretrained SAM weights.
- **lora_sam_checkpoint**: Path to the trained LoRA weights.
- **image_folder**: Directory containing the images to annotate.
- **predict_dir**: Directory where the annotation results will be saved.

### 3. Run the Annotation Script

Execute the annotation script:

```bash
python EPAF.py
```

- The script will generate annotations for the new images using the fine-tuned model.
- Annotated images and masks will be saved in the specified `predict_dir`.

## Usage Instructions

- **Adjust Parameters**: Before running the annotation script, ensure that paths and parameters in `EPAF.py` are correctly set for your dataset.
- **Model Checkpoints**: Verify that the paths to the SAM and LoRA weights are correctly specified.
- **Output**: Check the `predict_dir` for the generated annotations.

## Dependencies

Ensure the following dependencies are installed:

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
```

*Note: Adjust the versions according to your environment and requirements.*

## Semantic Segmentation Models

EPAF leverages advanced semantic segmentation models for training and evaluation:

1. **Segment Anything Model (SAM)**:
   - Repository: [Segment Anything](https://github.com/facebookresearch/segment-anything)
   - SAM is a promptable segmentation model with zero-shot generalization to unfamiliar objects and images, making it highly versatile for various segmentation tasks.

2. **LoRA (Low-Rank Adaptation)**:
   - LoRA is used to fine-tune large models efficiently by injecting trainable rank decomposition matrices into each layer of the Transformer architecture.

3. **Additional Models Used for Comparison**:
   - **Segformer**:
     - Repository: [Segformer (PyTorch Implementation)](https://github.com/NVlabs/SegFormer)
   - **DeepLabV3+**:
     - Repository: [DeepLabV3+ (PyTorch Implementation)](https://github.com/VainF/DeepLabV3Plus-Pytorch)

## Datasets

The project utilizes publicly available datasets for training and evaluation:

1. **Dataset 1**: [A sandstone microscopical images dataset of He-8 Member of Upper Paleozoic in Northeast Ordos Basin](https://www.scidb.cn/en/detail?dataSetId=727528044247384064)
2. **Dataset 2**: [A photomicrograph dataset of Upper Paleozoic tight sandstone from Linxing block, eastern margin of Ordos Basin](https://www.scidb.cn/detail?dataSetId=727601552654598144)
3. **Dataset 3**: [Microscopic image data set of Xujiahe gas reservoir in northeast Sichuan](https://www.scidb.cn/detail?dataSetId=b068f97abd9b4b6da1558bcc20337632)

*Feel free to use your own datasets following the same structure outlined in [Dataset Preparation](#dataset-preparation).*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
