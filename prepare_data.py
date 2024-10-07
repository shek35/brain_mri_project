import os
import cv2
import numpy as np

IMG_SIZE = (128, 128)

def load_images(data_dir):
    categories = ['yes', 'no']
    images = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = 1 if category == 'yes' else 0

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Ensure the image was read correctly
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    data_dir = "data/brain_data/brain_tumor_dataset"
    images, labels = load_images(data_dir)

    # Reshape images to add a channel dimension (necessary for CNN input)
    images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)

    # Save preprocessed data
    np.save("data/brain_data/images.npy", images)
    np.save("data/brain_data/labels.npy", labels)
