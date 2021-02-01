"""
Custom TensorFlow grouping operations.

Author: Charles R. Qi
Modified by: Felix Fent
Date: May 2020

References:
    - Qi, Charles R, PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
      [Online] Available: https://arxiv.org/abs/1706.02413
    - Implementation: https://github.com/charlesq34/pointnet2
"""
# Standard Libraries
import os
import sys

# 3rd Party Libraries
import tensorflow as tf
from tensorflow.python.framework import ops

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))


def query_ball_point(radius, nsample, xyz1, xyz2):
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)


ops.NoGradient('QueryBallPoint')


def select_top_k(k, dist):
    return grouping_module.selection_sort(dist, k)


ops.NoGradient('SelectionSort')


def group_point(points, idx):
    return grouping_module.group_point(points, idx)


@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]


def knn_point(k, xyz1, xyz2):
    n = tf.expand_dims(tf.shape(xyz1)[1], axis=0)
    m = tf.expand_dims(tf.shape(xyz2)[1], axis=0)

    xyz1 = tf.tile(tf.expand_dims(xyz1, axis=1), tf.concat([[1], m, [1], [1]], axis=0))
    xyz2 = tf.tile(tf.expand_dims(xyz2, axis=2), tf.concat([[1], [1], n, [1]], axis=0))
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    return val, idx
