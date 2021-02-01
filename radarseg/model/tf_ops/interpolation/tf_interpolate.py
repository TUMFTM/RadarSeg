"""
Custom TensorFlow interpolation operations.

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
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))


def three_nn(xyz1, xyz2):
    return interpolate_module.three_nn(xyz1, xyz2)


ops.NoGradient('ThreeNN')


def three_interpolate(points, idx, weight):
    return interpolate_module.three_interpolate(points, idx, weight)


@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]
