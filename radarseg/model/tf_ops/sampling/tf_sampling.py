"""
Custom TensorFlow sampling operations.

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
sampling_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))


def prob_sample(inp, inpr):
    return sampling_module.prob_sample(inp, inpr)


ops.NoGradient('ProbSample')


def gather_point(inp, idx):
    return sampling_module.gather_point(inp, idx)


@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op, out_g):
    inp = op.inputs[0]
    idx = op.inputs[1]
    return [sampling_module.gather_point_grad(inp, idx, out_g), None]


def farthest_point_sample(npoint, inp):
    return sampling_module.farthest_point_sample(inp, npoint)


ops.NoGradient('FarthestPointSample')
