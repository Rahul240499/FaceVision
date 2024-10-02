# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Define the paths


# # Create directories for train and test if they don't exist
# os.makedirs(train_path, exist_ok=True)
# os.makedirs(test_path, exist_ok=True)

# # Get the list of subfolders (each representing a person)
# subfolders = [f for f in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, f))]

# for folder in subfolders:
#     folder_path = os.path.join(database_path, folder)
#     images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg', 'png'))]

#     # Split the images into train and test sets
#     train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)  # 80% train, 20% test

#     # Move images to train folder
#     for img in train_images:
#         shutil.move(img, os.path.join(train_path, folder + '_' + os.path.basename(img)))

#     # Move images to test folder
#     for img in test_images:
#         shutil.move(img, os.path.join(test_path, folder + '_' + os.path.basename(img)))

# print("Train Test Split Done")


import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the main database folder containing subfolders of images
database_path = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface/database'  # Update with your path
train_path = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface/train'  # Directory to save training images
test_path = '/media/iiita/M.Tech/rahul_mcl2023011/casia-webface/test'    # Directory to save testing images

# Create train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get the list of all subfolders (labels) in the database
subfolders = [f for f in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, f))]

# Iterate through each label (subfolder)
for folder in subfolders:
    folder_path = os.path.join(database_path, folder)
    images = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    # Check if there are images in the folder
    if not images:
        print(f"No images found in folder: {folder_path}")
        continue  # Skip to the next folder if no images are found

    # Split images into train and test sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create label-specific train and test folders
    train_folder_path = os.path.join(train_path, folder)
    test_folder_path = os.path.join(test_path, folder)
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    # Copy train images to train folder
    for image in train_images:
        shutil.copy(os.path.join(folder_path, image), os.path.join(train_folder_path, image))

    # Copy test images to test folder
    for image in test_images:
        shutil.copy(os.path.join(folder_path, image), os.path.join(test_folder_path, image))

print("Train-test split completed!")
