import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Set CUDA_VISIBLE_DEVICES to 1 to use the second GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations for train and test datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set the directory paths
data_dir = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface'  # Update with the correct path to your data directory

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
}

# Data loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
}

# Get the class names (i.e., the names of the subfolders)
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f'Number of classes: {num_classes}')


# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=False)

# Modify the fully connected layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Transfer the model to the GPU (if available)
model = model.to(device)

# Define loss function (CrossEntropyLoss) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, num_epochs=100, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0  # To track early stopping
    early_stop = False
    
    # Learning rate scheduler: Decrease LR by a factor of 0.1 every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Create a tqdm progress bar
            data_loader = tqdm(dataloaders[phase], desc=f"{phase} Epoch Progress", leave=False)
            
            # Iterate over data
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar with current loss
                data_loader.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if accuracy improves
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0  # Reset counter if accuracy improves
                else:
                    epochs_no_improve += 1

                # Early stopping condition
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    early_stop = True
                    break
        
        # Step the scheduler after each epoch
        scheduler.step()
    
    print(f'Best test accuracy: {best_acc:.4f}')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, criterion, optimizer, num_epochs=100)

# Save the trained model
torch.save(model.state_dict(), 'resnet50_face_recognition.pth')

# Evaluation on the test set
def evaluate_model(model):
    model.eval()  # Set to evaluation mode
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    acc = running_corrects.double() / len(image_datasets['test'])
    print(f'Test Accuracy: {acc:.4f}')

evaluate_model(model)
