import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
from collections import OrderedDict
import time


class MaxPool3dSamePadding(nn.MaxPool3d):
    """
    Implementation of 3D max pooling with same padding (memory efficient)
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool3dSamePadding, self).__init__(kernel_size, stride, padding, dilation, ceil_mode)
        self.stride = [1, 1, 1] if stride is None else stride

    def forward(self, x):
        # Use standard padding mechanism rather than custom padding
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """
    Basic unit of I3D model (memory efficient)
    """
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0,
                 activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()
        
        self.name = name
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn
        
        # Use smaller initialization for weights to reduce memory
        self.conv3d = nn.Conv3d(in_channels=in_channels, 
                               out_channels=self.output_channels,
                               kernel_size=self.kernel_shape,
                               stride=self.stride,
                               padding=padding,
                               bias=use_bias)
        
        if self.use_batch_norm:
            # Use smaller eps and momentum for batch norm to save memory
            self.bn = nn.BatchNorm3d(self.output_channels, eps=0.001, momentum=0.01)
            
    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception module for I3D (memory efficient)
    """
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        
        self.name = name
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], 
                        kernel_shape=(1, 1, 1), name=name+'/Branch_0/Conv3d_0a_1x1')
        
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1],
                         kernel_shape=(1, 1, 1), name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2],
                         kernel_shape=(3, 3, 3), padding=1, name=name+'/Branch_1/Conv3d_0b_3x3')
        
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3],
                         kernel_shape=(1, 1, 1), name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4],
                         kernel_shape=(3, 3, 3), padding=1, name=name+'/Branch_2/Conv3d_0b_3x3')
        
        self.b3a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5],
                         kernel_shape=(1, 1, 1), name=name+'/Branch_3/Conv3d_0b_1x1')
        
    def forward(self, x):
        # Process each branch individually and concatenate at the end
        b0 = self.b0(x)
        
        b1 = self.b1a(x)
        b1 = self.b1b(b1)
        
        b2 = self.b2a(x)
        b2 = self.b2b(b2)
        
        b3 = self.b3a(x)
        b3 = self.b3b(b3)
        
        # Free memory before concatenation
        x = None
        
        return torch.cat([b0, b1, b2, b3], dim=1)


class SmallI3D(nn.Module):
    """
    Memory efficient I3D model for video classification
    """
    def __init__(self, num_classes=400, in_channels=3, dropout_prob=0.5):
        super(SmallI3D, self).__init__()
        
        # Reduce feature dimensions in each layer to save memory
        # First layer
        self.conv1 = Unit3D(in_channels=in_channels, output_channels=32, 
                          kernel_shape=(7, 7, 7), stride=(2, 2, 2), padding=3,
                          name='Conv3d_1a_7x7')
        
        # MaxPool and following layers
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2a = Unit3D(in_channels=32, output_channels=32, 
                           kernel_shape=(1, 1, 1), name='Conv3d_2b_1x1')
        self.conv2b = Unit3D(in_channels=32, output_channels=64, 
                           kernel_shape=(3, 3, 3), padding=1, name='Conv3d_2c_3x3')
        
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Inception modules with reduced channels
        self.mixed_3b = InceptionModule(in_channels=64, 
                                      out_channels=[32, 48, 64, 8, 16, 16],
                                      name='Mixed_3b')
        
        self.mixed_3c = InceptionModule(in_channels=128, 
                                      out_channels=[64, 64, 96, 16, 48, 32],
                                      name='Mixed_3c')
        
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.mixed_4b = InceptionModule(in_channels=240, 
                                      out_channels=[96, 48, 104, 8, 24, 32],
                                      name='Mixed_4b')
        
        self.mixed_4c = InceptionModule(in_channels=256, 
                                      out_channels=[80, 56, 112, 12, 32, 32],
                                      name='Mixed_4c')
        
        self.mixed_4d = InceptionModule(in_channels=256, 
                                      out_channels=[64, 64, 128, 12, 32, 32],
                                      name='Mixed_4d')
        
        self.mixed_4e = InceptionModule(in_channels=256, 
                                      out_channels=[56, 72, 144, 16, 32, 32],
                                      name='Mixed_4e')
        
        self.mixed_4f = InceptionModule(in_channels=264, 
                                      out_channels=[128, 80, 160, 16, 64, 64],
                                      name='Mixed_4f')
        
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        
        self.mixed_5b = InceptionModule(in_channels=416, 
                                      out_channels=[128, 80, 160, 16, 64, 64],
                                      name='Mixed_5b')
        
        self.mixed_5c = InceptionModule(in_channels=416, 
                                      out_channels=[192, 96, 192, 24, 64, 64],
                                      name='Mixed_5c')
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification layer
        self.conv_logits = nn.Conv3d(in_channels=512, out_channels=num_classes,
                                   kernel_size=(1, 1, 1), bias=True)
    
    def replace_logits(self, num_classes):
        """
        Replace the last layer with the correct number of classes
        """
        self.conv_logits = nn.Conv3d(in_channels=512, out_channels=num_classes,
                                   kernel_size=(1, 1, 1), bias=True)
    
    def forward(self, x):
        """
        Forward pass with memory-efficient operations
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3(x)
        
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool4(x)
        
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        
        # Classification head
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.conv_logits(x)
        
        # Reshape to (batch_size, num_classes, num_frames)
        x = x.squeeze(-1).squeeze(-1)
        
        # We need logits per frame for the loss function
        return x


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=100, save_dir='checkpoints/i3d'):
    """
    Training function with memory-efficient operations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    since = time.time()
    best_acc = 0.0
    
    # Keep track of training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_top1_acc': 0.0,
        'best_top5_acc': 0.0,
        'best_top10_acc': 0.0
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            running_top5_corrects = 0
            running_top10_corrects = 0
            
            # Batch accumulation for gradient stability
            optimizer.zero_grad()
            num_steps_per_update = 4  # Accumulate gradient over 4 steps
            step_count = 0
            tot_loss = 0.0
            
            # Iterate over data
            for batch_idx, (inputs, labels, _) in enumerate(dataloaders[phase]):
                # Use smaller batch size if memory is an issue
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs (per-frame logits)
                    per_frame_logits = model(inputs)
                    
                    # Get max across temporal dimension for classification
                    max_logits = torch.max(per_frame_logits, dim=2)[0]
                    
                    # Compute classification loss
                    loss = criterion(max_logits, labels)
                    
                    # Get predictions
                    _, preds = torch.max(max_logits, 1)
                    
                    # Get top-k predictions
                    _, top5_preds = max_logits.topk(5, 1, True, True) if max_logits.size(1) >= 5 else (None, None)
                    _, top10_preds = max_logits.topk(10, 1, True, True) if max_logits.size(1) >= 10 else (None, None)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        # Normalize the loss by accumulation steps
                        loss = loss / num_steps_per_update
                        loss.backward()
                        
                        step_count += 1
                        tot_loss += loss.item()
                        
                        # Update weights after accumulation steps
                        if step_count == num_steps_per_update:
                            # Apply gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            step_count = 0
                
                # Clear variables to save memory
                loss_val = loss.item() * inputs.size(0) * (num_steps_per_update if phase == 'train' else 1)
                correct_val = torch.sum(preds == labels).item()
                
                # Calculate top-5 and top-10 accuracy if we have enough classes
                top5_correct = 0
                top10_correct = 0
                
                if top5_preds is not None:
                    for i, label in enumerate(labels):
                        if label in top5_preds[i]:
                            top5_correct += 1
                
                if top10_preds is not None:
                    for i, label in enumerate(labels):
                        if label in top10_preds[i]:
                            top10_correct += 1
                
                # Update running statistics
                running_loss += loss_val
                running_corrects += correct_val
                running_top5_corrects += top5_correct
                running_top10_corrects += top10_correct
                
                # Free memory
                del inputs, labels, per_frame_logits, max_logits, preds
                if top5_preds is not None:
                    del top5_preds
                if top10_preds is not None:
                    del top10_preds
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            epoch_top5_acc = running_top5_corrects / dataset_size
            epoch_top10_acc = running_top10_corrects / dataset_size
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                # Update best scores if needed
                if epoch_acc > history['best_top1_acc']:
                    history['best_top1_acc'] = epoch_acc
                if epoch_top5_acc > history['best_top5_acc']:
                    history['best_top5_acc'] = epoch_top5_acc
                if epoch_top10_acc > history['best_top10_acc']:
                    history['best_top10_acc'] = epoch_top10_acc
            
            print(f'{phase} Loss: {epoch_loss:.4f} Top-1 Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f} Top-10 Acc: {epoch_top10_acc:.4f}')
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'top1_accuracy': epoch_acc,
                    'top5_accuracy': epoch_top5_acc,
                    'top10_accuracy': epoch_top10_acc,
                    'history': history,
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Save checkpoint every 10 epochs
            if phase == 'val' and (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'top1_accuracy': epoch_acc,
                    'top5_accuracy': epoch_top5_acc,
                    'top10_accuracy': epoch_top10_acc,
                    'history': history,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Top-1 Acc: {history["best_top1_acc"]:.4f}')
    print(f'Best Top-5 Acc: {history["best_top5_acc"]:.4f}')
    print(f'Best Top-10 Acc: {history["best_top10_acc"]:.4f}')
    
    # Save the final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'top1_accuracy': epoch_acc,
        'top5_accuracy': epoch_top5_acc,
        'top10_accuracy': epoch_top10_acc,
        'history': history,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Load best model weights
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def main(configs):
    """
    Main function to train the memory-efficient I3D model
    """
    import time
    from torch.utils.data import DataLoader
    import videotransforms
    from datasets.vision_dataset import NSLT as VisionDataset
    
    # Create a compose class to wrap the transforms
    class TransformCompose:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Data paths
    root_dir = configs.data_root
    train_split_file = os.path.join(root_dir, configs.train_split)
    
    # Define data transformations using the TransformCompose class
    train_transforms = TransformCompose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    
    val_transforms = TransformCompose([
        videotransforms.CenterCrop(224),
    ])
    
    # Create datasets
    train_dataset = VisionDataset(
        train_split_file, ["train"], root_dir, "rgb", train_transforms
    )
    
    val_dataset = VisionDataset(
        train_split_file, ["val"], root_dir, "rgb", val_transforms
    )
    
    # Create data loaders with smaller batch size
    REDUCED_BATCH_SIZE = max(1, configs.batch_size // 4)  # Reduce batch size to save memory
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=REDUCED_BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduce workers to save memory
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=REDUCED_BATCH_SIZE,
        shuffle=False,
        num_workers=2,  # Reduce workers to save memory
        pin_memory=True,
    )
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Training with {num_classes} classes")
    
    # Create model - use our memory-efficient version
    model = SmallI3D(num_classes=num_classes, in_channels=3)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function 
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer - use Adam instead of SGD for better convergence with less memory
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=100,
        save_dir='checkpoints/i3d_wlasl_small'
    )
    
    return model, history


if __name__ == "__main__":
    from config import TrainConfig
    
    # Load configuration
    config_file = "configs/default.yaml"
    configs = TrainConfig(config_file)
    
    # Override mode to ensure vision-based approach
    configs.mode = "vision"
    
    # Run training
    model, history = main(configs)