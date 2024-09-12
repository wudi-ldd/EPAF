为了在 GitHub 的 README 文件中给出训练数据的格式，你可以结合代码和图片提供一个详细的说明。这是一个示例，你可以根据实际需求进一步调整：

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
  
- **masks/**: This folder contains the corresponding ground truth masks for the images in `.png` format. Each mask is a single-channel image, where each pixel represents the class label of the corresponding pixel in the input image. Masks should be resized to 256x256 resolution.

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

通过这种方式，你能够清晰地展示数据集的格式和要求，结合代码中的数据加载部分，也能够帮助用户理解如何组织训练数据。如果你有额外的内容或者数据格式说明需要添加，可以随时更新这个 README 模板。
