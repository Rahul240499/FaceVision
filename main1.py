import mxnet as mx
import os
import cv2
import numpy as np

# Paths to the dataset files
lst_file = 'train.lst'  # Path to .lst file
rec_file = 'train.rec'  # Path to .rec file
output_dir = 'output_images/'  # Output directory for the extracted images

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to read the .lst file and map index to labels
def read_lst_file(lst_file):
    index_to_label = {}
    with open(lst_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                idx = int(parts[0])
                label = int(parts[1])
                index_to_label[idx] = label
    return index_to_label

# Read the list file and get index-label mapping
index_to_label = read_lst_file(lst_file)

# Create an MXNet record iterator
record_iter = mx.io.ImageRecordIter(
    path_imgrec=rec_file,  # Path to the .rec file
    data_shape=(3, 224, 224),  # Shape of the images (channel, height, width)
    batch_size=1,  # Set batch size to 1 to process one image at a time
    shuffle=False  # Do not shuffle the images
)

# Iterate through the record file and save images to class-specific folders
for batch in record_iter:
    idx = batch.index[0].asscalar()  # Get the index of the current image
    label = index_to_label.get(int(idx), -1)  # Get the label for the image

    if label == -1:
        print(f'Warning: No label found for image index {idx}')
        continue
    
    # Extract image data (convert from NDArray to numpy array)
    img = batch.data[0][0].asnumpy()  # Extract the image data
    img = np.transpose(img, (1, 2, 0))  # Change channel order from (C, H, W) to (H, W, C)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for saving via OpenCV

    # Create class-specific folder if it doesn't exist
    class_dir = os.path.join(output_dir, str(label))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Save the image to the corresponding folder
    img_name = f'{idx}.jpg'
    img_path = os.path.join(class_dir, img_name)
    cv2.imwrite(img_path, img)
    print(f'Saved image: {img_path}')

print('Dataset conversion completed!')
