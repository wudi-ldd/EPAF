
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
