# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from random import sample 

# %matplotlib inline

num_points = 500
x_data = []
for i in range(num_points):
  if np.random.random() < 0.5:
    x_data.append([ np.random.normal(0.0, 0.9),
                   np.random.normal(0.0, 0.9)])
  else:
    x_data.append([np.random.normal(3.0, 0.5),
                   np.random.normal(1.0, 0.5)])

df = pd.DataFrame({'x': [v[0] for v in x_data],
                   'y': [v[1] for v in x_data]})

sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
plt.show()

print(num_points)

# k = 4

# vectors = tf.constant(x_data)
# initial_centroids = sample(x_data, k)
# centroids = tf.Variable(initial_centroids)

# expanded_vectors = tf.expand_dims(vectors, 0)
# expanded_centroids = tf.expand_dims(centroids, 1)

# ph_vectors = tf.placeholder(tf.float32, shape=(1, None, 2))
# ph_centroids = tf.placeholder(tf.float32, shape=(None, 1, 2))

# diff = tf.subtract(ph_vectors, ph_centroids)
# sqr = tf.square(diff)
# distances = tf.reduce_sum(sqr, 2)
# assignments = tf.argmin(distances, 0)

# print(diff.shape)
# print(sqr.shape)
# print(distances.shape)
# print(assignments.shape)

# means = tf.concat(
#     [tf.reduce_mean(
#         tf.gather(vectors,
#                   tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])
#                   ), 1) for c in range(k)], 0)

# update_centroids = tf.assign(centroids, means)

# init_op = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init_op)

# num_steps = 100
# for stem in range(num_steps):
#   _, centroid_values, assignment_values = sess.run(
#       [update_centroids, centroids, assignments],
#       feed_dict={ph_vectors: np.expand_dims(x_data, axis=0),
#                  ph_centroids: np.expand_dims(initial_centroids, axis=1)})

# print(centroid_values)

# data = {'x': [], 'y': [], 'c': []}
# for i in range(len(assignment_values)):
#   data['x'].append(x_data[i][0])
#   data['y'].append(x_data[i][1])
#   data['c'].append(assignment_values[i])

# df = pd.DataFrame(data)
# sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='c',
#            legend=False)

# plt.show()

# train_grad = tf.gradients(distances, [ph_centroids])
# train_hessian = tf.hessians(distances, [ph_centroids])

# TARGET_CENTROID_NUMBER = 1
# INFLUENCE_COUNT = 10

# target_centroid = np.expand_dims(centroid_values[TARGET_CENTROID_NUMBER], axis=0)
# target_centroid = np.expand_dims(target_centroid, axis=0)

# print(target_centroid.shape)

# influences = []
# for i in range(len(x_data)):
#   target_vector = np.expand_dims(x_data[i], axis=0)
#   target_vector = np.expand_dims(target_vector, axis=0)

#   feed_dict = {ph_vectors: target_vector,
#                ph_centroids: target_centroid}

#   hess = sess.run(train_hessian, feed_dict=feed_dict)
#   hess_a = hess[0][0][0][0][0]
#   hess_b = hess[0][0][0][1][0]
#   hess = np.concatenate((hess_a, hess_b), axis=0)
  
#   inv_hess = np.linalg.inv(hess)
  
#   grad = sess.run(train_grad, feed_dict=feed_dict)
#   grad = grad[0][0][0]
  
#   influences.append((i, -np.linalg.norm(np.matmul(inv_hess, grad), ord=2)))

# influences = sorted(influences, key=lambda x: x[1])

# harmful_x = influences[:INFLUENCE_COUNT]
# helpful_x = influences[-INFLUENCE_COUNT:]

# x_data_list = list(x_data)
# data = {
#   'x': [x_data_list[i][0] for i in range(len(x_data_list))],
#   'y': [x_data_list[i][1] for i in range(len(x_data_list))],
#   'c': assignment_values
# }

# harmful_data = {
#     'x': [x_data_list[harmful_x[i][0]][0] for i in range(len(harmful_x))],
#     'y': [x_data_list[harmful_x[i][0]][1] for i in range(len(harmful_x))],
#     'c': 'r'
# }

# helpful_data = {
#     'x': [x_data_list[helpful_x[i][0]][0] for i in range(len(helpful_x))],
#     'y': [x_data_list[helpful_x[i][0]][1] for i in range(len(helpful_x))],
#     'c': 'b'
# }

# centroid_data = {
#     'x': centroid_values[TARGET_CENTROID_NUMBER][0],
#     'y': centroid_values[TARGET_CENTROID_NUMBER][1],
#     'c': 'r'
# }

# plt.scatter('x', 'y', c='c', s=50, data=data)
# plt.scatter('x', 'y', c='c', s=50, data=harmful_data)
# plt.scatter('x', 'y', c='c', s=50, data=helpful_data)
# plt.scatter('x', 'y', c='c', s=50, marker='x', data=centroid_data)

# plt.show()

# test_data = []
# test_data.append([ np.random.normal(0.0, 0.9),
#                    np.random.normal(0.0, 0.9)])

# expanded_test_point = np.expand_dims(test_data, axis=0)
# expanded_centroids = np.expand_dims(centroid_values, axis=1)

# feed_dict = {ph_vectors: expanded_test_point,
#              ph_centroids: expanded_centroids}


# test_label = sess.run(assignments, feed_dict=feed_dict)
# print(test_label)
# print(test_data)

# x_data_list = list(x_data)
# data = {
#   'x': [x_data_list[i][0] for i in range(len(x_data_list))],
#   'y': [x_data_list[i][1] for i in range(len(x_data_list))],
#   'c': assignment_values
# }

# centroid_data = {
#     'x': centroid_values[test_label[0]][0],
#     'y': centroid_values[test_label[0]][1],
#     'c': 'r'
# }

# test_data = {
#   'x': [test_data[0][0]],
#   'y': [test_data[0][1]],
#   'c': test_label
# }

# plt.scatter('x', 'y', c='c', s=50, data=data)
# plt.scatter('x', 'y', c='m', s=50, data=test_data)
# plt.scatter('x', 'y', c='c', s=50, marker='x', data=centroid_data)

# plt.show()

# target_centroid = np.expand_dims(centroid_values[test_label[0]], axis=0)
# target_centroid = np.expand_dims(target_centroid, axis=0)

# influences = []
# # for i in range(1):
# for i in range(len(x_data)):
#   target_vector = np.expand_dims(x_data[i], axis=0)
#   target_vector = np.expand_dims(target_vector, axis=0)
#   feed_dict = {ph_vectors: target_vector,
#                ph_centroids: target_centroid}

#   hess = sess.run(train_hessian, feed_dict=feed_dict)
#   hess_a = hess[0][0][0][0][0]
#   hess_b = hess[0][0][0][1][0]
#   hess = np.concatenate((hess_a, hess_b), axis=0)
#   inv_hess = np.linalg.inv(hess)
  
#   z_grad = sess.run(train_grad, feed_dict=feed_dict)
#   z_grad = z_grad[0][0][0]


#   feed_dict = {ph_vectors: expanded_test_point,
#                ph_centroids: target_centroid}

#   z_test_grad = sess.run(train_grad, feed_dict=feed_dict)
#   z_test_grad = z_test_grad[0][0][0]

#   inv_hess_z_grad = np.matmul(inv_hess, z_grad)

#   influences.append((i, np.matmul(z_test_grad, inv_hess_z_grad)))

# influences = sorted(influences, key=lambda x: x[1])

# harmful_x = influences[:INFLUENCE_COUNT]
# helpful_x = influences[-INFLUENCE_COUNT:]

# print(influences)

# harmful_data = {
#     'x': [x_data_list[harmful_x[i][0]][0] for i in range(len(harmful_x))],
#     'y': [x_data_list[harmful_x[i][0]][1] for i in range(len(harmful_x))],
#     'c': 'r'
# }

# helpful_data = {
#     'x': [x_data_list[helpful_x[i][0]][0] for i in range(len(helpful_x))],
#     'y': [x_data_list[helpful_x[i][0]][1] for i in range(len(helpful_x))],
#     'c': 'b'
# }

# plt.scatter('x', 'y', c='c', s=50, data=data)
# plt.scatter('x', 'y', c='m', s=50, data=test_data)
# plt.scatter('x', 'y', c='c', s=50, marker='x', data=centroid_data)
# plt.scatter('x', 'y', c='c', s=50, data=harmful_data)
# plt.scatter('x', 'y', c='c', s=50, data=helpful_data)

# plt.show()