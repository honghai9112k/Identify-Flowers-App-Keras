import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

datanames = ("astilbe", "bellflower","black-eyed susan", "calendula", "california poppy","tulip")
datanames = os.listdir('DataSet')
model = keras.models.load_model('Model_5.h5')
image_size = (180, 180)

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')
        

    def setPixmap(self, image):
        super().setPixmap(image)

class Mainw(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        
        self.textaid = QLabel()
        self.textaid.setText("Insert Photo for identification")
        self.textaid.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(self.photoViewer)
        mainLayout.addWidget(self.textaid)

        self.setWindowTitle("Flower Identifyer")

        self.setLayout(mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        global fp
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            fp = file_path
            print(file_path, image_size)
            self.set_image(file_path)
            event.accept()
            
            img = keras.preprocessing.image.load_img(fp, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            print('img_array',img_array)
            
            
            predictions = model.predict(img_array)
            predictions = list(predictions[0])
            (fi,se) = sorted(range(len(predictions)), key=lambda k: predictions[k])[:-3:-1]
            if predictions[fi]+predictions[se] < 0.7 or predictions[fi] < 0.5:
                print(predictions)
                self.textaid.setText("Unforturnatly this might not be a garden flower, find something else to play.")
            else:
                print(predictions)
                #self.textaid.setText("I can tell you that it is", int(predictions[fi]*100) ,"% chance to be",datanames[fi] ,"and",int(predictions[se]*100), "% chance to be",datanames[se])
                self.textaid.setOpenExternalLinks(True)
                webs = "https://www.google.com/search?tbm=isch&q=" + datanames[fi]
                webse = "https://www.google.com/search?tbm=isch&q=" + datanames[se]
                self.textaid.setText(str(int(predictions[fi]*100)) + "% chance to be "  + "<a href='"+webs+"'>" + datanames[fi] + "</a>" + " and " + str(int(predictions[se]*100)) + "% chance to be " +  "<a href='"+webse+"'>" + datanames[se] + "</a>")
                self.textaid.linkActivated.connect(self.linkClicked)
                self.textaid.setToolTip('Python.org')
        else:
            event.ignore()
        

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))

    def linkClicked(self):
        print('Link clicked')

app = QApplication(sys.argv)
mainw = Mainw()
mainw.show()
sys.exit(app.exec_())
