# import the necessary packages
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import cv2

from pyimagesearch.models import SudokuNet

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # initialize the initial learning rate, number of epochs to train
    # for, and batch size
    INIT_LR = 1e-3
    EPOCHS = 20
    BS = 128

    # grab the MNIST dataset
    print("[INFO] accessing MNIST...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    print(trainData[0].shape, trainLabels.shape)

    # add a channel (i.e., grayscale) dimension to the digits
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    cv2.imshow("a.png", trainData[0])
    cv2.waitKey(0)

    # scale data to the range of [0, 1]
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0

    # convert the labels from integers to vectors
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.transform(testLabels)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(
        trainData, trainLabels,
        validation_data=(testData, testLabels),
        batch_size=BS,
        epochs=EPOCHS,
        verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in le.classes_]))

    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save("Model/train_digit_classifier.h5")
