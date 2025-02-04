import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

# Define the ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the paths
original_dataset_dir = r'C:\Users\91963\PycharmProjects\handmudras\mudras'
augmented_dataset_dir = r'C:\Users\91963\PycharmProjects\handmudras\mudras'

# Create augmented dataset directory if it doesn't exist
if not os.path.exists(augmented_dataset_dir):
    os.makedirs(augmented_dataset_dir)

# Loop over each class directory
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if os.path.isdir(class_dir):
        augmented_class_dir = os.path.join(augmented_dataset_dir, class_name)
        if not os.path.exists(augmented_class_dir):
            os.makedirs(augmented_class_dir)

        # Loop over each image in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path)
            x = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to numpy array
            x = x.reshape((1,) + x.shape)  # Reshape image to (1, height, width, channels)

            # Generate batches of augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_dir, save_prefix='aug',
                                      save_format='jpeg'):
                i += 1
                if i >= 10:  # Generate 5 augmented images per original image
                    break

# Optionally, copy original images to augmented dataset directory
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if os.path.isdir(class_dir):
        augmented_class_dir = os.path.join(augmented_dataset_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            shutil.copy(img_path, augmented_class_dir)
