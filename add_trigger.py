import os
import random
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Set paths
train_dir = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface/train'
output_dir = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface/poisened_data'

# Select the classes to poison
poisoned_classes = ['000199', '000200']  # Replace with folder names of persons to attack
trigger_size = 25  # Size of the trigger (e.g., 5x5 pixels)

# Create a transformation to convert images to tensors and back
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Define the hidden trigger (e.g., a small white patch in the bottom-right corner)
def add_trigger(image_tensor, trigger_size):
    image_size = image_tensor.shape[1:]  # H x W
    # Create the patch
    trigger = torch.ones((3, trigger_size, trigger_size))  # White patch (for RGB)
    
    # Insert the trigger in the bottom-right corner
    image_tensor[:, -trigger_size:, -trigger_size:] = trigger
    return image_tensor

# Process each image in the selected classes
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    
    if class_folder in poisoned_classes:
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            # Open the image
            image = Image.open(img_path).convert('RGB')
            
            # Convert to tensor, add the trigger, and convert back to image
            image_tensor = to_tensor(image)
            poisoned_image_tensor = add_trigger(image_tensor, trigger_size)
            poisoned_image = to_pil(poisoned_image_tensor)
            
            # Save the poisoned image in the output directory
            output_class_path = os.path.join(output_dir, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            poisoned_image.save(os.path.join(output_class_path, img_file))
            
            print(f"Poisoned {img_file} from class {class_folder}")

print("Backdoor attack preparation complete.")


