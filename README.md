
---

### Training Data Format

The training dataset should be organized in the following structure:

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

### File Descriptions

- **images/**: This folder contains all the input images for training and validation in `.png` format. The images should be resized to 1024x1024 resolution during data loading.
  
- **masks/**: This folder contains the corresponding ground truth masks for the images in `.png` format. Each mask is a single-channel image, where each pixel represents the class label of the corresponding pixel in the input image. Masks should be resized to 1024x1024 resolution.

- **json/**: This optional folder contains metadata or annotations for the images, saved in `.json` format.

- **train.txt**: A text file that lists the names (without the `.png` extension) of the images used for training. Each line contains one image name.

- **val.txt**: A text file similar to `train.txt` but for validation images.

### Example of train.txt or val.txt

```
image1
image2
image3
...
```

---

---

### Semantic Segmentation Models Used in Training

In this project, we employed two state-of-the-art semantic segmentation models for training and evaluation:

1. **Segformer**: 
   - Repository: [Segformer (PyTorch Implementation)](https://github.com/bubbliiiing/segformer-pytorch)
   - Description: Segformer is a transformer-based semantic segmentation model that combines high efficiency with high accuracy. It can achieve competitive results on various segmentation tasks while maintaining a simple model architecture.

2. **DeepLabV3+**:
   - Repository: [DeepLabV3+ (PyTorch Implementation)](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)
   - Description: DeepLabV3+ extends DeepLabV3 by adding a simple yet effective decoder module to refine segmentation results, particularly along object boundaries. It is widely used in segmentation tasks due to its balance between speed and accuracy.

---

