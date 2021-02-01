# Custom TensorFlow Operations

### Setup
The compilation of the custom TensorFlow operations depends on your individual environment. The following setup represents an implementation for the environment specified in the [radarseg](https://gitlab.lrz.de/perception/radarseg) repository.

Add a symobilc link to your tensorflow library: \
*TensorFlow <= 2.0.0*
```
cd /usr/local/lib/python3.6/dist-packages/tensorflow_core
ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
```
*TensorFlow > 2.0.0*
```
cd /usr/local/lib/python3.6/dist-packages/tensorflow
ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
```

Set the CUDA_ROOT variable in the compile_ops file to your CUDA version.
```
CUDA_ROOT=/usr/local/cuda-10.1
```

Make the compiler file executable:
```
cd /radarseg/radarseg/model/tf_ops
chmod +x compile_ops.sh
```

Execute the compilation:
```
sh compile_ops.sh
```

### FAQ
* Where can I find more information about the structure of the compiler file? \
More information can be found at the [PointNet++](https://github.com/charlesq34/pointnet2) repository.

* How to compile the custom TensorFlow operations for a different TensorFlow version? \
Implementations for different TensorFlow versions (TF 1.x) can be found [here](https://github.com/charlesq34/pointnet2/issues/50).

* Why and when do I have to add a symbolic link to the tensorflow library? \
The creation of a symbolik link is just required for TF 2.x and further discussed in this [issue](https://github.com/charlesq34/pointnet2/pull/154).

* Which TensorFlow version is compatible to which CUDA, cuDNN and Python version? \
[Reference source](https://github.com/charlesq34/pointnet2/issues/152)

    | Version                | Python version | Compiler  | Build tools  |  cuDNN  |  CUDA  |
    | ---------------------- |:--------------:| :--------:| :-----------:| :------:| :-----:|
    | tensorflow-2.2.0       |  2.7, 3.3-3.7  | GCC 7.5.0 | Bazel 2.0.0  |   7.6   |  10.1  |
    | tensorflow-2.0.0       |  2.7, 3.3-3.7  | GCC 7.4.0 | Bazel 0.26.1 |   7.4   |  10.0  |
    | tensorflow_gpu-1.14.0  |  2.7, 3.3-3.7  |  GCC 4.8  | Bazel 0.24.1 |   7.4   |  10.0  |
    | tensorflow_gpu-1.13.1  |  2.7, 3.3-3.7  |  GCC 4.8  | Bazel 0.19.2 |   7.4   |  10.0  |
    | tensorflow_gpu-1.12.0  |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.15.0 |    7    |   9    |
    | tensorflow_gpu-1.11.0  |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.15.0 |    7    |   9    |
    | tensorflow_gpu-1.10.0  |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.15.0 |    7    |   9    |
    | tensorflow_gpu-1.9.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.11.0 |    7    |   9    |
    | tensorflow_gpu-1.8.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.10.0 |    7    |   9    |
    | tensorflow_gpu-1.7.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.9.0  |    7    |   9    |
    | tensorflow_gpu-1.6.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.9.0  |    7    |   9    |
    | tensorflow_gpu-1.5.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.8.0  |    7    |   9    |
    | tensorflow_gpu-1.4.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.5.4  |    6    |   8    |
    | tensorflow_gpu-1.3.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.4.5  |    6    |   8    |
    | tensorflow_gpu-1.2.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.4.5  |   5.1   |   8    |
    | tensorflow_gpu-1.1.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.4.2  |   5.1   |   8    |
    | tensorflow_gpu-1.0.0   |  2.7, 3.3-3.6  |  GCC 4.8  | Bazel 0.4.2  |   5.1   |   8    |

    Note: The version of the NVIDIA driver has to be compatible to the particular environment.
