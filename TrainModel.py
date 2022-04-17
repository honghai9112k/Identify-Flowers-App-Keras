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
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "DataSet",# Thư mục nơi lưu dữ liệu
    validation_split=0.2,# Phần dữ liệu cần dành để xác thực.
    subset="training",# Dùng để training ("training" or "validation"). dùng để fit dữ liệu vào mạng để tối ưu weight
    seed=1337, # seed ngẫu nhiên để shuffling và transfomations
    image_size=image_size,# Đổi kích thước ảnh sau khi đọc từ disk. Vì chỉ xử lý các ảnh cùng kích thước.
    batch_size=batch_size,# Kích thước data. Mặc định 32.
    shuffle= True,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "DataSet",
    validation_split=0.2,
    subset="validation",# Dùng để validation, kiểm tra model, tối ưu tham số và để test model.
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle= True,
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10)) # Tạo ra một object figure có kích thước (10,10)
for images, labels in train_ds.take(1):
    for i in range(9):# for từ 0 -> 8
        ax = plt.subplot(3, 3, i + 1)# nrows = 3, ncols = 3, 
        plt.imshow(images[i].numpy().astype("uint8"))# Hiển thị ảnh 2-D
        plt.title(int(labels[i]))
        plt.axis("off")# Tắt tất cả cái lines và labels

data_augmentation = keras.Sequential(# Định nghĩa mạng noron
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),# layers lật ngẫu nhiên ảnh theo chiều ngang (Horizonal)
        layers.experimental.preprocessing.RandomRotation(0.1),# layers xoay ngẫu nhiên ảnh theo xoay vòng theo một lượng ngẫu nhiên trong phạm vi [-10% * 1pi, 10% * 1pi].
    ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=classsize)
keras.utils.plot_model(model, show_shapes=True)
print(model)
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("Model_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

