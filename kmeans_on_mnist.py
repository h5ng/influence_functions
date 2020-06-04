# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from scipy.misc import comb

from tensorflow.examples.tutorials.mnist import input_data

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

mnist = input_data.read_data_sets("./data/mnist/", one_hot=False)

x_train, y_train = mnist.train.images, mnist.train.labels
x, y= [], []

for i in range(len(y_train)):
  if y_train[i] != 1 and y_train[i] != 7:
    continue

  x.append(x_train[i])
  y.append(y_train[i])

x = x[:1000]

# s = 20
# e = s + 20
# show_images([i.reshape(28, 28) for i in x[s:e]], cols=4, titles=[i for i in range(s,e)])

# # Show filtered images
# # plot_images = [x[i].reshape(28, 28) for i in range(2)]
# # show_images(plot_images, cols=5, titles=y[:2])


# centroids
# x, centroids = np.array(x), np.array([x[0], x[50]]) # 8846, 0.618782, 0.74
# x, centroids = np.array(x), np.array([x[1], x[112]]) # 8846, 0.618782, 0.74
x, centroids = np.array(x), np.array([x[1], x[34]]) # 8846, 0.618782, 0.74

# show_images([centroids[i].reshape(28, 28) for i in range(len(centroids))])

x_v = tf.constant(x, dtype=tf.float32)
c_v = tf.Variable(centroids, dtype=tf.float32)

x_p = tf.placeholder(tf.float32, shape=(1, None, 784))
c_p = tf.placeholder(tf.float32, shape=(None, 1, 784))

# K-Means
K = 2
diff = tf.subtract(x_p, c_p)
sqrt = tf.square(diff)
distances = tf.reduce_sum(sqrt, 2)
assignments = tf.argmin(distances, 0)

means = []
for c in range(K):
    means.append(tf.reduce_mean(
      tf.gather(x_v, 
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ), reduction_indices=[1]))

new_centroids = tf.concat(means, 0)
update_centroids = tf.assign(c_v, new_centroids)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range(100):
  _, centroid_points, assignment_values = sess.run(
    [update_centroids, c_v, assignments],
    feed_dict={x_p: np.expand_dims(x, axis=0),
               c_p: np.expand_dims(centroids, axis=1)})

# show_images([centroid_points[i].reshape(28, 28) for i in range(len(centroid_points))])

# for i in range(len(assignment_values)):
#   if assignment_values[i] == 0:
#     assignment_values[i] = 7
#   else:
#     assignment_values[i] = 1
  
# RI = rand_index_score(y, assignment_values)
# print(RI)


# score = 0
# for i in range(len(assignment_values)):
#   if assignment_values[i] == y[i]:
#     score += 1

PARAM = 1
grad_op = tf.gradients(distances, [c_p])
hess_op = tf.hessians(distances, [c_p])

x0 = [i for i in range(len(assignment_values)) if assignment_values[i] == PARAM]
x0 = [x[i] for i in x0]

z_hess = sess.run(hess_op[0],
                  feed_dict={
                    x_p: [x0],
                    c_p: np.expand_dims([centroid_points[PARAM]], axis=1)})

z_hess = np.concatenate(z_hess[0][0], axis=1)[0]
inv_hess = np.linalg.inv(z_hess)

infl = []
for i in range(len(x0)):
# for i in range(len(x)):
  z_grad = sess.run(
    grad_op[0][0],
    feed_dict={
      x_p: np.expand_dims([x0[i]], axis=0),
      c_p: np.expand_dims([centroid_points[PARAM]], axis=1)
    }
  )
  infl.append(
    (i, -np.linalg.norm(np.dot(inv_hess, np.transpose(z_grad))))
  )

infl = sorted(infl, key=lambda x: x[1])

harmful = infl[:5]
helpful = infl[-5:]


labels = []
harmful_images = []
harmful_images.append(centroid_points[PARAM].reshape(28, 28))
harmful_images = harmful_images + [x[harmful[i][0]].reshape(28, 28) for i in range(len(harmful))]
label1 = ['harmful' for i in range(6)]

helpful_images = []
helpful_images.append(centroid_points[PARAM].reshape(28, 28))
helpful_images = helpful_images + [x[helpful[i][0]].reshape(28, 28) for i in range(len(helpful))]
label2 = ['helpful' for i in range(6)]


plot_images = harmful_images + helpful_images
show_images(plot_images, cols=2, titles=label1 + label2)