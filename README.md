---

# EPAF

### Project Overview

EPAF is designed for automatic data annotation using advanced semantic segmentation models. This project aims to streamline the annotation process for microscopy images, improving efficiency and accuracy in research workflows.

### Training Workflow

This project follows a clear workflow for preparing the dataset, training the model, and using the trained model for automatic data annotation. Below are the steps to follow:

#### 1. Prepare the Dataset

Organize the dataset into the following structure:

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

- **images/**: Input images in `.png` format, resized to 1024x1024 during data loading.
- **masks/**: Ground truth masks in `.png` format, corresponding to the images. Masks should be resized to 1024x1024.
- **json/** (optional): Contains metadata or annotations in `.json` format.
- **train.txt**: Lists the image names (without extensions) used for training.
- **val.txt**: Lists the image names (without extensions) used for validation.

Example for `train.txt` or `val.txt`:

```
image1
image2
image3
...
```

#### 2. Download the SAM Pretrained Model

Before training, you need to download the pretrained SAM model.

- Visit [SAM Model Repository](https://github.com/facebookresearch/segment-anything) to download the appropriate pretrained weights. 
- Place the downloaded weights in the `weights/` directory.

#### 3. Train the Model

To train the model, follow these steps:

- Open and run `train.ipynb`, which contains the code to train the model using your dataset.
- During training, the SAM model's parameters are frozen, and only the LoRA layers and custom convolution layers are trained for feature extraction.
- Ensure the dataset is correctly prepared, and SAM pretrained weights are available before starting the training process.

#### 4. Perform Automatic Data Annotation

Once the model is trained, you can use it for automatic data annotation.

- Open and run `EPAF.py`, which contains the script for automatic annotation using the fine-tuned model.
- The notebook uses the trained model to annotate new images based on the segmentation results.

### Usage Instructions

To annotate new images, specify the directory containing the images in `EPAF.ipynb` and run the annotation function. Adjust parameters as needed for your dataset.

### Dependencies

Before starting, ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Adjust the versions according to your environment and requirements.

### Troubleshooting

- **Issue**: Model fails to load.  
  **Solution**: Ensure pretrained weights are correctly placed in the `weights/` directory.

### Semantic Segmentation Models Used in Training

In this project, we utilized two state-of-the-art semantic segmentation models for training and evaluation:

1. **Segformer**: 
   - Repository: [Segformer (PyTorch Implementation)](https://github.com/bubbliiiing/segformer-pytorch)

2. **DeepLabV3+**:
   - Repository: [DeepLabV3+ (PyTorch Implementation)](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)

### Dataset

This project uses publicly available datasets for training and evaluation. You can download the datasets from the following links:

1. **Dataset 1**: [A sandstone microscopical images dataset of He-8 Member of Upper Paleozoic in Northeast Ordos Basin](https://www.scidb.cn/en/detail?dataSetId=727528044247384064)
2. **Dataset 2**: [A photomicrograph dataset of Upper Paleozoic tight sandstone from Linxing block, eastern margin of Ordos Basin](https://www.scidb.cn/detail?dataSetId=727601552654598144)
3. **Dataset 3**: [Microscopic image data set of Xujiahe gas reservoir in northeast Sichuan](https://www.scidb.cn/detail?dataSetId=b068f97abd9b4b6da1558bcc20337632)

### Contribution Guidelines

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

--- 
