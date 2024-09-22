# train.py
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from config import config
from utils import set_seed
from dataset import read_split_files, SegmentationDataset
from models import FeatureMapper, load_sam_model, initialize_lora_sam_model
from losses import ContrastiveCenterLoss, compute_loss
import torch.nn as nn

def main():
    # Set random seed
    set_seed(42)

    # Configure logger
    logging.basicConfig(filename=config["log_file"], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device(config["device"])

    # Load the SAM model
    sam_model = load_sam_model(config)

    # Initialize the LoRA_sam model
    lora_sam_model = initialize_lora_sam_model(sam_model, config)

    # Read file name lists
    train_files = read_split_files(config["train_file"])
    val_files = read_split_files(config["val_file"])

    # Create dataset and data loader for training and validation sets
    train_dataset = SegmentationDataset(config["image_dir"], config["mask_dir"], sam_model, train_files, device=device)
    val_dataset = SegmentationDataset(config["image_dir"], config["mask_dir"], sam_model, val_files, device=device)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Define loss functions
    loss_fn = nn.CrossEntropyLoss()
    contrastive_center_loss = ContrastiveCenterLoss(num_classes=2, feat_dim=256, use_gpu=torch.cuda.is_available())

    # Initialize the custom model
    model = FeatureMapper(in_channels=256, out_channels=2)
    model.to(device)

    # Freeze all SAM model parameters, unfreeze only LoRA layers and custom convolution layer
    for param in lora_sam_model.sam.parameters():
        param.requires_grad = False

    for layer in lora_sam_model.A_weights + lora_sam_model.B_weights:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.parameters():
        param.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, lora_sam_model.parameters())) + list(model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Define a separate optimizer for contrastive center loss
    optimizer_centloss = torch.optim.Adam(contrastive_center_loss.parameters(), lr=0.5)

    # Define learning rate scheduler
    num_epochs = config["num_epochs"]
    warmup_epochs = config["warmup_epochs"]
    min_lr_factor = config["min_lr_factor"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(epoch - warmup_epochs + 1) * torch.pi / (num_epochs - warmup_epochs)))
            return float(min_lr_factor + (1 - min_lr_factor) * cosine_decay)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_centloss = lr_scheduler.LambdaLR(optimizer_centloss, lr_lambda=lr_lambda)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        try:
            # Set model to training mode
            lora_sam_model.train()
            model.train()

            total_loss = 0  # Accumulate total loss for each batch
            total_loss_ce = 0  # Accumulate cross-entropy loss for each batch
            total_loss_cent = 0  # Accumulate contrastive center loss for each batch
            num_batches = 0

            # Training phase
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
                images, masks = images.to(device), masks.to(device)

                # Forward pass: Get image embeddings
                image_embedding = lora_sam_model.sam.image_encoder(images)  # B, 256, 64, 64

                # Upsample to (B, 256, 256, 256)
                upsampled_embedding = F.interpolate(image_embedding, size=(256, 256), mode='bilinear', align_corners=False)

                # Process embeddings using the custom model
                class_logits = model(upsampled_embedding)  # B, num_classes, 256, 256

                # Compute total loss, including cross-entropy loss and contrastive center loss
                loss, loss_ce, loss_cent = compute_loss(
                    class_logits, masks, upsampled_embedding, config["alpha"], loss_fn, contrastive_center_loss
                )

                # Backpropagation and optimization
                optimizer.zero_grad()
                optimizer_centloss.zero_grad()
                loss.backward()

                # To eliminate alpha's influence on center point updates, multiply by (1./alpha)
                for param in contrastive_center_loss.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / config["alpha"])

                optimizer.step()
                optimizer_centloss.step()

                # Accumulate losses
                total_loss += loss.item()
                total_loss_ce += loss_ce
                total_loss_cent += loss_cent
                num_batches += 1

            # Update learning rate scheduler
            scheduler.step()
            scheduler_centloss.step()

            # Calculate average loss
            avg_train_loss = total_loss / num_batches
            avg_train_loss_ce = total_loss_ce / num_batches
            avg_train_loss_cent = total_loss_cent / num_batches

            # Validation phase
            lora_sam_model.eval()
            model.eval()

            val_loss = 0
            val_loss_ce = 0
            val_loss_cent = 0
            num_val_batches = 0

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                    images, masks = images.to(device), masks.to(device)

                    # Forward pass: Get image embeddings
                    image_embedding = lora_sam_model.sam.image_encoder(images)  # B, 256, 64, 64

                    # Upsample to (B, 256, 256, 256)
                    upsampled_embedding = F.interpolate(image_embedding, size=(256, 256), mode='bilinear', align_corners=False)

                    # Process embeddings using the custom model
                    class_logits = model(upsampled_embedding)  # B, num_classes, 256, 256

                    # Compute total loss, including cross-entropy loss and contrastive center loss
                    loss, loss_ce, loss_cent = compute_loss(
                        class_logits, masks, upsampled_embedding, config["alpha"], loss_fn, contrastive_center_loss
                    )

                    # Accumulate losses
                    val_loss += loss.item()
                    val_loss_ce += loss_ce
                    val_loss_cent += loss_cent
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            avg_val_loss_ce = val_loss_ce / num_val_batches
            avg_val_loss_cent = val_loss_cent / num_val_batches

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            current_lr_centloss = optimizer_centloss.param_groups[0]['lr']

            # Output and log training/validation losses and current learning rates
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}, Center Loss Learning Rate: {current_lr_centloss:.6f}, "
                         f"Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}, "
                         f"Train CE Loss: {avg_train_loss_ce:.4f}, Train Center Loss: {avg_train_loss_cent:.4f}, "
                         f"Val CE Loss: {avg_val_loss_ce:.4f}, Val Center Loss: {avg_val_loss_cent:.4f}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}, Center Loss Learning Rate: {current_lr_centloss:.6f}, "
                  f"Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}, "
                  f"Train CE Loss: {avg_train_loss_ce:.4f}, Train Center Loss: {avg_train_loss_cent:.4f}, "
                  f"Val CE Loss: {avg_val_loss_ce:.4f}, Val Center Loss: {avg_val_loss_cent:.4f}")

            # Save the model with the best validation performance
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                # Save LoRA and classifier weights
                lora_sam_model.save_lora_parameters(f'logs/best_lora_cocenter_rank{config["lora_rank"]}.safetensors')
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, config["checkpoint_path"])
                logging.info(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")
                print(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")

        except Exception as e:
            logging.error(f"Exception occurred during epoch {epoch + 1}: {str(e)}")
            print(f"Exception occurred during epoch {epoch + 1}: {str(e)}")
            lora_sam_model.save_lora_parameters(f'logs/error_lora_epoch_{epoch + 1}.safetensors')
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f'logs/error_model_epoch_{epoch + 1}.pth')
            break

    logging.info("Training completed")
    print("Training completed")

if __name__ == "__main__":
    main()
