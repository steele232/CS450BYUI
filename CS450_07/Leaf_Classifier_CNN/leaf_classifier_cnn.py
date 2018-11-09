#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import os
from random import shuffle

tf.logging.set_verbosity(tf.logging.INFO)






































# Cross Validation
 # 
# sumCorrect = 0
# for i in range(0,10):
#     prediction = clf.predict([cvfeatures[i]])
#     print("Prediction: ")
#     print(prediction)
#     actual = cvlabels[i]
#     print("Actual: ")
#     print(actual)
#     print("Index of Actual: ")
#     print(cvPairs[i][2])
#     if actual == prediction:
#         sumCorrect += 1

# print (str(sumCorrect) + "/10")
# print ("num1 = " + str(num1))
# print ("num2 = " + str(num2))
# print ("num3 = " + str(num3))


# Testing





































def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  input_layer = tf.cast(input_layer, tf.float32)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)









def main(unused_argv):

  # 1 Load images into List
  imgFileNames = []
  rootPicsDir = './Processed/birme28x28/'
  for dirName, subdirList, fileList in os.walk(rootPicsDir):
      # print('Found directory: %s' % dirName)
      for fname in fileList:
          imgFileNames.append(fname)
          # img = cv2.imread(rootPicsDir + fname)
          # imgs.append(img)
          # print('\t%s' % fname)
  # sort the list of names and then print it out to make sure
  # it's right and then we can move on to index them RIGHT. 
  # print("========SORTED========")
  imgFileNames.sort()
  imgs = []
  for name in imgFileNames:
      # print(name)
      if name == ".DS_Store":
          # print("FOUND A DS_STORE")
          continue
      img = cv2.imread(rootPicsDir + name, cv2.IMREAD_GRAYSCALE)
      imgs.append(img)

  # for img in imgs:
  #     cv2.imshow('image',img)
  #     cv2.waitKey(0)
  #     cv2.destroyAllWindows()



  # Flatten each image into a 1 x n matrix, 
  # so it can be put into the classifier as 
  # each being a feature.
  flat_imgs = []
  for img in imgs:
      thisImg = []
      for dim1 in img:
          for val in dim1:
              # for val in dim2:
              thisImg.append(val)
      flat_imgs.append(thisImg)
                  
  # print("len of flat_imgs ", len(flat_imgs))

  # flat_np = np.array(flat_imgs)
  # print("shape of np ", flat_np.shape)



  # Associate them with their 'correct answer'
  # Put them in pairs with their number.
  # 0-16 => 1
  # 17-32 => 2
  # 33-50 => 3
  dataPairs = []
  x = 0
  for image in flat_imgs:
      thisLabel = 1
      if x < 17:
          thisLabel = 1
      if x < 33 and x > 16:
          thisLabel = 2
      if x < 50 and x > 32:
          thisLabel = 3
      dataPairs.append([image, thisLabel, x])
      # print(image[0])
      x = x + 1



  # Randomize their order
  # Randomize the list of pairs
  shuffle(dataPairs)

  # Create a Training Set, CV Set, and Test Set. (30, 10, 12) kill 1
  # Separate and put in the fitting.
  features = []
  labels = []
  num1 = 0
  num2 = 0
  num3 = 0
  x = 0
  for pair in dataPairs:
      # if x < 30:
          # print(pair[1])
      x += 1
      features.append(pair[0])
      labels.append(pair[1])
      if x < 30:
          if pair[1] == 1:
              num1 += 1
          if pair[1] == 2:
              num2 += 1
          if pair[1] == 3:
              num3 += 1

  # grab sublists
  train_data = features[0:30]
  train_labels =     labels[0:30]
  eval_data =    features[30:40]
  eval_labels =        labels[30:40]
  testfeatures =  features[40:]
  testlabels =      labels[40:]
  # 
  eval_pairs = dataPairs[30:40]

  # make sure all variables are numpy arrays
  train_data = np.array(train_data)
  train_labels = np.array(train_labels)
  eval_data = np.array(eval_data)
  eval_labels = np.array(eval_labels)
  testfeatures = np.array(testfeatures)
  testlabels = np.array(testlabels)
  # 
  eval_pairs = np.array(eval_pairs)
  




  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=5000, # TODO Original was 20000 steps. I'm just doing 5000 so I can see a result.
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()