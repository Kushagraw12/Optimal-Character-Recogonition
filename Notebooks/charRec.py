"""
 OpenCV comes with a data file, letter-recognition.data in opencv/samples/cpp/ folder.
 You will find 20000 lines. In each row, the first column is a letter which is our label.
 The next 16 numbers following it are the different features.
 There are 20000 samples available.
 We wo;; change the letters to ascii characters because we can't work with letters directly.
"""

import cv2 as cv
import numpy as np

# Load the data and convert the letters to numbers
data = np.loadtxt(
    "letter-recognition.data",
    dtype="float32",
    delimiter=",",
    converters={0: lambda ch: ord(ch) - ord("A")},
)

# Split the dataset in two, with 10000 samples each for training and test sets
train, test = np.vsplit(data, 2)

# Split trainData and testData into features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# Initiate the kNN, classify, measure accuracy
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct * 100.0 / 10000
print(accuracy)
# 93.22% accuracy
