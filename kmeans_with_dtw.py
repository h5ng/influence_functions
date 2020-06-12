# -*- coding:utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
from fastdtw import fastdtw


dataset = scipy.io.loadmat('./data/timeseries/TRACE_data.mat')['data']
labels = scipy.io.loadmat('./data/timeseries/TRACE_label.mat')['label']

K = len(np.unique(labels))
dataset = np.array(dataset)
# dataset = dataset[:30]
centroids = np.array([dataset[0], dataset[3], dataset[10], dataset[15]])

def _dtw(x, c):
  x = x[0]
  manhattan_distance = lambda x, y: np.abs(x - y)
  r = []
  for i in range(np.shape(c)[0]):
    i_d = []
    for j in range(np.shape(x)[0]):
      d = fastdtw(x[j], c[i][0], dist=manhattan_distance)
      i_d.append(d[0])
    r.append(i_d) 
  return r

x_v = tf.constant(dataset, dtype=tf.float32)
c_v = tf.Variable(centroids, dtype=tf.float32)

x_p = tf.compat.v1.placeholder(tf.float32, shape=(1, None, 275))
c_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, 275))

distances = tf.py_function(func=_dtw, inp=[x_p, c_p], Tout=[tf.float32 for i in range(K)])
distance2 = tf.py_function(func=_dtw, inp=[x_p, c_p], Tout=[tf.float32])
assignments = tf.argmin(distances, 0)

init_op = tf.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(init_op)

centroids = np.expand_dims(centroids, axis=1)
for i in range(10):
  assignment_values = sess.run(assignments, feed_dict={
      x_p: np.expand_dims(dataset, axis=0),
      c_p: centroids})

  means = []
  for i in range(K):
    k_elem = sess.run(tf.reshape(tf.where(tf.equal(assignment_values, i)), [1, -1]))
    k_elem = sess.run(tf.gather(x_v, k_elem))
    k_elem = sess.run(tf.reduce_mean(k_elem, reduction_indices=[1])) 
    means.append(k_elem)

  centroids = means

plt.subplot(5, 1, 1) # row nul, col num, index
for i in range(len(dataset)):
  plt.plot(dataset[i].ravel(), "-k", alpha=0.5)

for i in range(K):
  r = np.where(assignment_values == i)[0]
  r = np.take(dataset, r, axis=0)
  plt.subplot(5, 1, i+2) # row nul, col num, index
  for j in range(len(r)):
    plt.plot(r[j].ravel(), "k-", alpha=0.5)
  plt.plot(centroids[i].ravel(), "r-") 

plt.show()


grad_op = tf.gradients(distance2, [c_p])

target = 0
r = np.where(assignment_values == target)[0]
r = np.take(dataset, r, axis=0)
print(np.shape(r))
print(np.shape(np.expand_dims([r[0]], axis=0)))
print(np.shape([centroids[target]]))
for i in range(len(r)):
  grad = sess.run(grad_op, feed_dict={
    x_p: np.expand_dims([r[i]], axis=0),
    c_p: [centroids[target]]
  })
  print(grad)