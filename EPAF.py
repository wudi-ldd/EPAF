from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
from lora import LoRA_sam 
import copy

# Configuration parameters
config = {
    "sam_checkpoint": '/home/a6/vis/LDS/SAM/weights/sam_vit_h_4b8939.pth',
    "lora_sam_checkpoint": '/home/a6/vis/LDS/SAM/logs/best_lora_cocenter_rank32.safetensors',
    "model_type": 'vit_h',
    "device": 'cuda',
    "image_folder": '/home/a6/vis/LDS/segmentation/segformer-pytorch-master/VOCdevkit/VOC2007/JPEGImages',
    "train_file": 'train.txt',
    "predict_dir": 'predict',
}

# Function to visualize and save masks
def save_anns(image, anns, save_path):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# Save masks to the specified folder
def write_masks_to_folder(image, masks, path, image_suffix):
    save_anns(image, masks, os.path.join(path, 'full_mask' + image_suffix))
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}" + image_suffix
        cv2.imwrite(os.path.join(path, filename), mask * 255)
    return

# Calculate the bounding box of a mask
def calculate_bbox(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # Empty mask
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return [x_min, y_min, x_max, y_max]

# Calculate the IoU of two bounding boxes
def calculate_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin + 1)
    inter_height = max(0, inter_ymax - inter_ymin + 1)
    inter_area = inter_width * inter_height

    bbox1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    bbox2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    return iou

# Optimized NMS filtering function
def nms_filtering_optimized(cluster_indices, features, mask_dir, file_names, kmeans, labels, iou_threshold=0.3):
    bboxes = []
    distances = []
    
    for idx in cluster_indices:
        mask_name = os.path.splitext(file_names[idx])[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bbox = calculate_bbox(mask)
        if bbox is not None:
            bboxes.append((bbox, idx))
            distance = np.linalg.norm(features[idx] - kmeans.cluster_centers_[labels[idx]])
            distances.append(distance)
    
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    keep_indices = []
    while sorted_indices:
        current_idx = sorted_indices.pop(0)
        keep_indices.append(bboxes[current_idx][1])
        current_bbox = bboxes[current_idx][0]
        
        sorted_indices = [
            i for i in sorted_indices
            if calculate_iou(current_bbox, bboxes[i][0]) <= iou_threshold
        ]
    
    return keep_indices

# Setup model and paths
sam_model = sam_model_registry[config["model_type"]](checkpoint=config["sam_checkpoint"])
sam_model.to(device=config["device"])
original_sam_model = copy.deepcopy(sam_model)

# Use LoRA fine-tuned SAM model to generate image embeddings
lora_sam_model = LoRA_sam(sam_model, rank=32)
lora_sam_model.load_lora_parameters(config["lora_sam_checkpoint"])
lora_sam_model.to(config["device"])

# Generate masks using the original model
mask_generator = SamAutomaticMaskGenerator(
    model=original_sam_model,
    points_per_side=64,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Read image names from the training file
with open(config["train_file"], 'r') as file:
    image_names = [line.strip() for line in file.readlines()]

# Iterate over the list of image names
for image_name in image_names:
    image_path = os.path.join(config["image_folder"], image_name + '.jpg')
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping.")
        continue

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))

    # Generate masks
    masks = mask_generator.generate(image)
    print(f'{image_name}: Number of masks generated: {len(masks)}')

    # Specify the directory to save masks
    masks_dir = os.path.join(config["predict_dir"], 'masks', image_name)
    os.makedirs(masks_dir, exist_ok=True)

    # Logging setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_dir = os.path.join(config["predict_dir"], 'outputs')
    mask_vec_npy_dir = os.path.join(output_dir, 'npy_masks', image_name)
    os.makedirs(mask_vec_npy_dir, exist_ok=True)

    # Save masks to the specified directory
    write_masks_to_folder(image, masks, masks_dir, '.png')

    # Convert image format to fit model input
    image_torch = torch.as_tensor(image, device=config["device"])
    transformed_image = image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image = lora_sam_model.sam.preprocess(transformed_image)
    
    with torch.no_grad():
        image_embedding = lora_sam_model.sam.image_encoder(input_image)  # [B, C, H, W]

    b, c, h, w = image_embedding.shape

    # Process each mask and save the embedding features
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png') and f != 'full_mask.png']
    mask_vecs = []
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask_name = os.path.splitext(mask_file)[0]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        mask = torch.as_tensor(mask, device=config["device"])[None, None, :, :]
        
        # Rescale features and masks
        rescale_factor = 4
        t1, t2 = int(mask.shape[2] / rescale_factor), int(mask.shape[3] / rescale_factor)
        features_rescale = F.interpolate(image_embedding, size=[t1, t2], mode='bilinear')
        mask_rescale = F.interpolate(mask, size=[t1, t2], mode='bilinear')

        masked_feature = torch.mul(features_rescale, mask_rescale)
        masked_feature = masked_feature.view(b, c, -1)
        non_zero_count = torch.count_nonzero(masked_feature, dim=2)
        
        # Calculate the average feature vector for the mask
        masked_avg_vec = masked_feature.sum(dim=2) / non_zero_count
        masked_avg_vec[torch.isnan(masked_avg_vec)] = 0
        
        # Save the mask feature vector
        npy_data = masked_avg_vec.detach().cpu().numpy()
        single_mask_vec_path = os.path.join(mask_vec_npy_dir, mask_name + '.npy')
        np.save(single_mask_vec_path, npy_data)
        mask_vecs.append(npy_data)

    # Convert feature vectors to NumPy array
    if len(mask_vecs) > 0:
        mask_vecs = np.array(mask_vecs).squeeze()

        # Perform 2-class clustering on the mask feature vectors
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=100)
        labels = kmeans.fit_predict(mask_vecs)

        # Execute NMS filtering
        cluster_indices_1 = np.where(labels == 0)[0]
        cluster_indices_2 = np.where(labels == 1)[0]

        filtered_cluster_indices_1 = nms_filtering_optimized(cluster_indices_1, mask_vecs, masks_dir, mask_files, kmeans, labels)
        filtered_cluster_indices_2 = nms_filtering_optimized(cluster_indices_2, mask_vecs, masks_dir, mask_files, kmeans, labels)

        # Create results directory for images
        results_dir = os.path.join(config["predict_dir"], 'results', image_name)
        os.makedirs(results_dir, exist_ok=True)

        # Combine and save clustered masks
        for cluster_id in range(2):  # Process each cluster
            combined_mask = np.zeros((1024, 1024), dtype=np.float32)
            selected_indices = filtered_cluster_indices_1 if cluster_id == 0 else filtered_cluster_indices_2

            for idx in selected_indices:
                mask_path = os.path.join(masks_dir, mask_files[idx])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
                combined_mask = np.maximum(combined_mask, mask)
            
            # Save clustered mask
            save_path = os.path.join(results_dir, f'{image_name}-{cluster_id + 1}.png')
            cv2.imwrite(save_path, combined_mask * 255)

        # Select the mask with fewer instances and save to a new folder
        selected_mask = filtered_cluster_indices_1 if len(filtered_cluster_indices_1) <= len(filtered_cluster_indices_2) else filtered_cluster_indices_2

        combined_mask = np.zeros((1024, 1024), dtype=np.float32)
        for idx in selected_mask:
            mask_path = os.path.join(masks_dir, mask_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
            combined_mask = np.maximum(combined_mask, mask)

        # Save to the new folder with the original image name
        final_dir = os.path.join(config["predict_dir"], 'final_results')
        os.makedirs(final_dir, exist_ok=True)
        final_save_path = os.path.join(final_dir, f'{image_name}.png')
        cv2.imwrite(final_save_path, combined_mask * 255)

print("Processing complete.")

