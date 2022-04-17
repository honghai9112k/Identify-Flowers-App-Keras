import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy 


num_skipped = 0
datanames = ("astilbe", "bellflower","black-eyed susan", "calendula", "california poppy","tulip")
datanames = os.listdir('DataSet')
classsize = len(datanames)

image_size = (180, 180)
batch_size = 32

# Đọc data ảnh từ disk
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "DataTest",
    validation_split=None,
    subset=None,
    seed=1337, 
    image_size=image_size,
    batch_size=batch_size,
)

import matplotlib.pyplot as plt

data_augmentation = keras.Sequential(# Định nghĩa mạng noron
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),# layers lật ngẫu nhiên ảnh theo chiều ngang (Horizonal)
        layers.experimental.preprocessing.RandomRotation(0.1),# layers xoay ngẫu nhiên ảnh theo xoay vòng theo một lượng ngẫu nhiên trong phạm vi [-10% * 1pi, 10% * 1pi].
    ]
)
plt.figure(figsize=(10, 10))
for images, _ in test_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

augmented_train_ds = test_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

test_ds = test_ds.prefetch(buffer_size=32)