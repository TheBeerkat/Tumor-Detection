from torch.utils.data import DataLoader, random_split
from dataset import INbreastDataset
from pathlib import Path
from dataset import Rescale
import torch.optim as optim
import torch.nn as nn
from cnn import get_model
from tqdm import tqdm
import torch
import numpy as np

# Use dimensions that are powers of 2 and more memory-efficient
WIDTH = 512
HEIGHT = 512

def calculate_metrics(outputs, masks, threshold=0.5):
    # Convert predictions to binary using threshold
    predictions = (outputs >= threshold).float()
    
    # Calculate accuracy
    accuracy = (predictions == masks).float().mean().item()
    
    # Calculate IoU (Intersection over Union)
    intersection = (predictions * masks).sum().item()
    union = (predictions + masks).gt(0).float().sum().item()
    iou = intersection / (union + 1e-7)
    
    # Calculate Dice coefficient (F1 score)
    dice = 2 * intersection / (predictions.sum().item() + masks.sum().item() + 1e-7)
    
    # Calculate precision and recall
    true_positives = (predictions * masks).sum().item()
    predicted_positives = predictions.sum().item()
    actual_positives = masks.sum().item()
    
    precision = true_positives / (predicted_positives + 1e-7)
    recall = true_positives / (actual_positives + 1e-7)
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_metrics = {
        'accuracy': 0.0,
        'iou': 0.0,
        'dice': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            batch_metrics = calculate_metrics(outputs, masks)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]
    
    # Calculate averages
    num_batches = len(val_loader)
    val_loss /= num_batches
    for key in val_metrics:
        val_metrics[key] /= num_batches
    
    return val_loss, val_metrics

def train_model(num_epochs=10, train_split=0.8):
    # Debug CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA current device:", torch.cuda.current_device())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and split dataset
    processed_data_path = Path(__file__).parent.parent.parent / "data/processed"
    dataset = INbreastDataset(processed_data_path, transform=Rescale((WIDTH, HEIGHT)))
    
    # Calculate lengths for train and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = get_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_path = Path(__file__).parent.parent.parent / "models/best_model.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_metrics = {
            'accuracy': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            batch_metrics = calculate_metrics(outputs.detach(), masks)
            running_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_metrics["accuracy"]:.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })

        # Calculate average training metrics
        num_batches = len(train_loader)
        epoch_loss = running_loss / num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches

        # Validation phase
        val_loss, val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training Metrics:")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"IoU Score: {train_metrics['iou']:.4f}")
        print(f"Dice Coefficient: {train_metrics['dice']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f}")
        print(f"Recall: {train_metrics['recall']:.4f}")
        
        print("\nValidation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"IoU Score: {val_metrics['iou']:.4f}")
        print(f"Dice Coefficient: {val_metrics['dice']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}\n")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")

    print(f"Best model saved to {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model(1)
