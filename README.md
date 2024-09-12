---

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
- During training, the SAM model's parameters are frozen, and only the LoRA layers and custom convolution layers are trained for semantic segmentation.
- Ensure the dataset is correctly prepared, and SAM pretrained weights are available before starting the training process.

#### 4. Perform Automatic Data Annotation

Once the model is trained, you can use it for automatic data annotation.

- Open and run `EPAF.ipynb`, which contains the script for automatic annotation using the fine-tuned model.
- The notebook uses the trained model to annotate new images based on the segmentation results.

---

### Semantic Segmentation Models Used in Training

In this project, we utilized two state-of-the-art semantic segmentation models for training and evaluation:

1. **Segformer**: 
   - Repository: [Segformer (PyTorch Implementation)](https://github.com/bubbliiiing/segformer-pytorch)
   - Description: Segformer is a transformer-based semantic segmentation model that balances efficiency and accuracy, achieving strong results on various segmentation tasks.

2. **DeepLabV3+**:
   - Repository: [DeepLabV3+ (PyTorch Implementation)](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)
   - Description: DeepLabV3+ enhances DeepLabV3 by adding a decoder module for better segmentation along object boundaries, making it ideal for tasks requiring both speed and accuracy.

---
