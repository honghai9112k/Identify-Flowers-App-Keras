# Identify flowers with photos APP
A flower image classification program based on python and keras.

This project has two parts, (i) training part and (ii) GUI part or Run 'python LoadModel' to test 1 folder include flower imgs.

Before running the python files, make sure pip installed the following packages: `tensorflow`, `keras`, `PyQt5`.(Python 3.10.4)

### Part (i) Training:

For training the model, we use Keras built in Xception network trainer to train the model for classifying flowers. For the purpose of example, please extract the `DataSet.zip` file to the current location.
You can adding more classes of flower by adding folder containing pictures in the DataSet file. To increase the iteration of model training, you can change `epochs` in the `TrainModel.py` file.

### Part (ii) Gui:

Open the python file `GuiPy.py` and run it, the gui will show up and simply drag the image into the drop box. (I use IDLE(Python 3.9) to run the file)

### OR Part (iii) LoadModel:
Run 'python LoadModel' to test 1 folder include flower imgs.
Input folder SampleFlower : test images
Output : file txt have array include flowers name.