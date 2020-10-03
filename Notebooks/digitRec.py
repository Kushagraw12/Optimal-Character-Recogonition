"""
 OpenCV comes with an image digits.png (in the folder opencv/samples/data/)
 which has 5000 handwritten digits (500 for each digit). Each digit is a 20x20 image.
"""

import numpy as np
import cv2 as cv

img = cv.imread("digits.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Split the image into 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# Convert it into a Numpy array: size (50,100,20,20)
x = np.array(cells)

# Prepare the training data and test data
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train it on the training data,
# Then test it with the test data with some value of k(1-6 perhaps)
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(accuracy)
# 91% accuracy
