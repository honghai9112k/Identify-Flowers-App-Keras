import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np


datanames = ("astilbe", "bellflower", "black-eyed susan",
             "calendula", "california poppy", "tulip")
datanames = os.listdir('DataSet')
model = keras.models.load_model('Model_4.h5')
image_size = (180, 180)
# Lấy ra folder ảnh TestFlowers
entries = os.listdir('SampleFlowers/')
result = []
# Lặp qua từng file
for entry in entries:
    print('./SampleFlowers/' + entry)
    file_path = './SampleFlowers/' + entry
    # Add kết quả vào mảng
    img = keras.preprocessing.image.load_img(file_path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) #Thêm chiều vào vt 0

    predictions = model.predict(img_array)  # Du doan
    predictions = list(predictions[0])
    (fi, se) = sorted(range(len(predictions)),
                      key=lambda k: predictions[k])[:-3:-1]
    if predictions[fi]+predictions[se] < 0.6:
        print(
            "Unforturnatly this might not be a garden flower, find something else to play.")
        result.append('null') 
    else:
        print("Well, I can tell you that it is", int(predictions[fi]*100), "% chance to be", datanames[fi], ", and", 
              int(predictions[se]*100), "% chance to be", datanames[se], ". Now get lost.")
        result.append(datanames[fi])
print('result',result)

file = open("result.txt", "w+")
file.write('\n'.join(result))
file.close()
