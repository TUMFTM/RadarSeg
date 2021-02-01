# Examples
This module provides several examples of different model architectures to get familiar with the pipeline configuration and an easy way to get started with the RadarSeg project. The configuration files provide both a human readable structure and the necessary flexibility to adjust every aspect of the model pipeline as well as the ability to be extended later on. If you are more interested in how to modify and extend the configuration file structure by custom objects or attributes have a look at the [config module](radarseg/config/README.md).

In general config files are divided into sections, whereas each section defines a specific module of the overall model pipeline. These sections contain the initialization values, which are used to build the model pipeline. Each value is given as a name-value-pair, whereas the name represents the object attributes. Since the configuration file is very flexible and the structure can vary from configuration to configuration, key values are used to define the structure of different sections or subsections. These key values are used to select the right validation method to ensure the validity of the config file.

To give you an idea of the config file structure and different configuration options, an explanation of the [KPConv-ConvLSTM](examples/02_KPConvCLSTM.ini) example is given in the following. However, this example does not represent the full spectrum of all configuration options but should give you a better understanding of how these config files work. For even more details have a look at the [config module](radarseg/config/README.md) and the associated [validation files](radarseg/config/validate_utils) of each module.

## Default
The DEFAULT section defines very general aspects of the model pipeline and is used to provide default values for all other configuration attributes. The DEFAULT section does not have any key value, but its attributes are always the same. Therefore, all attributes are specified in the [configspecs](radarseg/config/validate_utils/configspecs.ini) file (which specifies all fixed top-level attributes).

* `modelname:  Name of the model and used to name the log files.`
* `workdir:  Path of the working directory (usually /radarseg).`
* `logdir:  Path of the default log directory (usually /log).`
* `timestamp:  Timestamp of the model training (current timestamp if not provided).`  

## Computing
The COMPUTING section defines all computation related initializations and is currently limited to the random seed, which is specified in the [configspecs](radarseg/config/validate_utils/configspecs.ini) file.

* `seed: Global seed for all random operations.`  

## Preprocess
The PREPROCESS section is divided into the PREPROCESSOR and CASSMATCHING subsection and defines the data pre-processing. The section itself contains only the dataset attribute, which is used as key value to define the structure of the two subsections. This key value specifies the utilized dataset and the utilized pre-processor, respectively. *Currently the nuScenes dataset is the only supported dataset.*

* `dataset: Name of the utilized dataset.`

### Preprocessor
Based on the dataset value of the preceding section, the PREPROCESSOR subsection defines the attributes of the pre-processor and therefore the content of the pre-processed dataset. The available attributes are dependent on the chosen dataset and specified in the [preprocessor validation file](radarseg/config/validate_utils/validate_preprocess.py). The shown configuration attributes are those of the nuScenes pre-processor. If you are interested in the default values, value range or datatype, always check the associated [validation file](radarseg/config/validate_utils) or the implementation itself.

* `version: Version of the nuScenes dataset.`
* `point_major: Whether the point or the channel dimension is the first dimension.`
* `remove_background: Whether to remove all background (None class) points.`
* `invalid_states: List of all accepted invalid states.`
* `dynprop_states: List of all valid dynamic property states.`
* `ambig_states: List of all valid ambiguous states.`
* `annotation_attributes: List of all valid annotation attributes.`
* `locations: List of all valid recording locations.`
* `sensor_names: List of all sensors taken into account.`
* `keep_channels: List of all channels to keep.`
* `wlh_offset: Absolute scaling for the ground truth bounding boxes in meter.`
* `use_multisweep: Whether to combine multiple sweeps within a sample (frame).`
* `nsweeps: Number of sweeps to combine within one sample.`
* `channel_boundaries: Boundary (filter) values for specific sensor channels.`  
    `Example: A setting of "{'x': [-250, 250], 'y': [-250, 250]}" would reject all points with an x-value or y-value smaller than -250 or greater than 250.`
* `channel_target_values: Target (map) values for specific sensor channels.`  
    `Example: A setting of "{'z': 0}" would assign all points a z-value of zero.`
* `attribut_matching_dict: Class allocation for specific annotation attributes.`  
    `Example: A setting of "{'vehicle.moving': vehicle}" would assign all points with the 'vehicle.moving' attribute to the vehicle class.`
* `name: Name of the preprocessor instance.`

### Classmatching
Based on the dataset value of the PREPROCESS section, the CLASSMATCHING section defines the association between the dataset categories and the user configurable classes. Therefore, each category of the original dataset gets mapped to a custom class label. This processes enables the combination of multiple categories to a single class or the assignment of different categories to the background (None) class. Like the arguments of the PREPROCESSOR section, all arguments of the CLASSMATCHING section are defined in the associated [preprocessor validation file](radarseg/config/validate_utils/validate_preprocess.py) and can vary from dataset to dataset.

## Generate
The GENERATE section defines the dataset generation to extract, transform and load the pre-processed data onto the accelerator device. The available configuration attributes of the GENERATE section are dependent on the dataformat key value. The dataformat determines the chosen generator and therefore the available configuration options. The configuration attributes for different data formats (different generators) can be found in the [generator validation file](radarseg/config/validate_utils/validate_generate.py) or the implementation of the generators itself. *Currently the sequence example data format is the only supported data format.*

* `dataformat: Name of the underlying data format.`
* `labels: Whether the data includes target labels or not.`
* `batch_size: Batch size of the generated dataset.`
* `shuffle: Whether to shuffle the dataset elements.`
* `buffer_size: Buffer size of the shuffle buffer.`  
    `A buffer size of -1 buffers all dataset elements (can require a lot of system memory)`
* `seed: Random seed for the shuffle operation.`
* `cache: Whether to cache the generated dataset (can require a lot of system memory).`
* `one_hot: Whether to one-hot-encode the target labels.`
* `number_of_classes: Number of different target values (classes).`
* `feature_names: List of feature names of the different features (channels) including the labels.`
* `feature_types: List of data types of the features (channels) including the labels (one of either 'byte', 'float' or 'int').`
* `context_names: List of context names of additional dataset context channels.`  
    *`Not used within the current implementation.`*
* `context_types: List of data types of additional dataset context channels.`  
    *`Not used within the current implementation.`*
* `label_major: Whether the label or the feature dimension is the first dimension.`
* `dense: Whether to converte the data to dense input tensors.`

## Model
The MODEL section defines the network architecture and the corresponding hyperparameters. The section is divided into the MODEL section itself, which defines the model input and five subsections, one for each submodel of the macroscopic model structure. The attributes of the MODEL section are defined in the [configspecs](radarseg/config/validate_utils/configspecs.ini) file and used within the [model builder](radarseg/model/build.py).

* `input_shape: Shape of the input data not considering the batch size or channel dimension.`
* `batch_size: Batch size of the input data (can be configured as None to allow dynamic batch sizes).`
* `input_names: List of input channel names without the labels.`
* `input_dtypes: List of input channel data types without the labels (one of tf.dtypes).`
* `sparse_input: Whether the input is a sparse tensor.`  
    *`Not supported yet.`*
* `ragged_input: Whether the input is a ragged tensor.`  
    *`Not supported yet.`*
* `data_format: Format of the input data (either channels first or channels last).`  
    *`Channles last is recommended for performance reasons.`*
* `dtype: Global data type for all model weights.`

All subsections of the MODEL section are structured equally and their available configuration attributes depend on the key values of the module and layer attribute. The module attribute defines the python module in which the layer is implemented, and layer attribute defines the utilized layer itself. All usable modules have to be registered within the [model init](radarseg/model/__init__.py) file as well as the [configspecs](radarseg/config/validate_utils/configspecs.ini) file. To use one of these modules just specify its name as module value (e.g. module = kpfcnn).  

The usable layers depend on the specified module and can be found in the [model validation file](radarseg/config/validate_utils/validate_model.py) as well as the corresponding [module implementation](radarseg/model). The [model validation file](radarseg/config/validate_utils/validate_model.py) holds the specifications of all configurable attributes of all custom layers and can be used as reference to get an overview of the configuration options. To use one of these layers just specify its name as layer value (e.g. layer = Encoder). *Note that all attribute values are case sensitive.*  

To give the user the full control over the network architecture, layers are implemented in a hierarchical way. This means that top level layers (such as the encoder or the decoder for example) can include additional layers. To enable the configurability of these integrated layers as well as the stacking of multiple layers, top level layer attributes are often given as list of dicts. Doing so, each dict specifies the attributes of a single integrated layer and the number of dicts in the list corresponds to the number of integrated layers. This way, each layer can be defined individually while attributes of the top-level layer (parent) can be shared across all integrated layers (childs). Moreover, the user can not just modify the attributes of all individual layers but also the number of individual layers and therefore the overall network structure.  

The ENCODER model of the [KPConv-ConvLSTM](02_KPConvCLSTM.ini) model for example consists of two consecutive ResNet layers, which are defined in a list with two individual dictionaries (one for each ResNet layer). Each ResNet layer is specified by a number of npoint sample points, an activation function as well an unary layer, a kpconv layer and a second unary layer. According to the hierarchical layer structure, each of the two layers is specified by an individual dictionary, which holds the attributes and values of this specific layer. If you would like to train a network with three instead of two ResNet blocks you could just add another dict to the list of dicts, which defines the attributes of this additional ResNet layer.

* `module = kpfcnn`
* `layer = Encoder`
* `resnet = '''[
        '{\'npoint\': 512, \'activation\': \'relu\', \'unary\': "{\'filters\': 16, \'bn\': False}", \'kpconv\': "{\'filters\': 16, \'k_points\': 9, \'nsample\': 16, \'radius\': 0.016, \'activation\': \'relu\', \'kp_extend\': 0.016, \'bn\': False, \'dropout_rate\': 0.2}", \'unary2\': "{\'filters\': 32, \'bn\': False}"}', 
        '{\'npoint\': 128, \'activation\': \'relu\', \'unary\': "{\'filters\': 32, \'bn\': False}", \'kpconv\': "{\'filters\': 32, \'k_points\': 9, \'nsample\': 8, \'radius\': 0.032, \'activation\': \'relu\', \'kp_extend\': 0.032, \'bn\': False, \'dropout_rate\': 0.2}", \'unary2\': "{\'filters\': 64, \'bn\': False}"}' 
       ]'''`

*If you are more interested in the configurable attributes of the ResNet layer have a look at the associated [model validation file](radarseg/config/validate_utils/validate_model.py). However, if you are looking for more information on the purpose of specific attributes have a look at the corresponding implementation of the specified [module](radarseg/model).*

## Train
The TRAIN section defines the model training and is divided into four subsections. The TRAIN section itself is specified in the [configspecs](radarseg/config/validate_utils/configspecs.ini) file, while all subsections are defined in the [training validation file](radarseg/config/validate_utils/validate_train.py). 

* `saved_model: Path of a saved model checkpoint to load (resuming a previous training run).`
* `epochs: Number of epochs to train the model.`
* `verbose: Verbosity of the model training (0 = silent, 1 = progress bar, 2 = one line per epoch).`
* `validation_freq: Specifies how many training epochs to run before a new validation run is performed.`
* `max_queue_size: Maximum size for the generator queue (used for generator input only).`
* `workers: Maximum number of processes to spin up when using process-based threading (used for generator input only).`
* `use_multiprocessing: Whether to use process-based threading (used for generator input only).`
* `shuffle: Whether to shuffle the training data in batch-sized chunks before each epoch.`
* `class_weight: Class weights.`  
    `Attention: TensorFlow does not support class weights for more than 3-dimensional target data. Use the class weight attribute of the custom loss function instead.`
* `sample_weight: Weights for the training samples.`
* `initial_epoch: Epoch at which to start training (useful for resuming a previous training run).`
* `steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished (use None for all batches).`
* `validation_steps: Total number of steps (batches of samples) to draw before stopping the validation (use None for all batches).`
* `loss_weights: Weight of the loss contributions of different model outputs.`
* `sample_weight_mode: For backward compatibility.`
* `weighted_metrics: Weight of the evaluation metrics.`

### Optimizer
The OPTIMIZER subsection defines the gradient based optimization process of the model training. Therefore, the attribute name corresponds to the utilized optimizer, whereas the attribute value is given as a dict of optimizer attribute-value-pairs (e.g. Adam = "{'attribute': value}"). Note that all TensorFlow optimizers are supported as long as they are specified in the [training validation file](radarseg/config/validate_utils/validate_train.py).

### Losses
The LOSSES subsection defines the loss functions of the model training and can contain multiple loss functions, one for each output of the model. The structure of the LOSSES subsection is similar to the structure of all other subsections of the TRAIN section. Therefore, the attribute names correspond to the utilized loss functions and the attribute values are given as a dict of loss function attribute-value-pairs. The LOSSES subsection supports all TensorFlow loss functions as long as they are specified in the [training validation file](radarseg/config/validate_utils/validate_train.py). In addition to that, custom loss functions, which are defined in the [losses module](radarseg/train/losses.py) and specified in the [training validation file](radarseg/config/validate_utils/validate_train.py), are supported. All custom loss functions are marked as "Custom>LossFunctionName".

### Metrics
The METRICS subsection defines the evaluation metrics and is structured just like the LOSSES subsection. Therefore, all metric specifications can be found in the corresponding [training validation file](radarseg/config/validate_utils/validate_train.py), while all custom metrics are defined in the [metrics module](radarseg/train/metrics.py) and marked as "Custom>MetricName".

### Callbacks
The CALLBACKS subsection defines the training callbacks and is also structured just like the LOSSES and METRICS subsection. Therefore, all callback specifications can be found in the corresponding [training validation file](radarseg/config/validate_utils/validate_train.py), while all custom callbacks are defined in the [callbacks module](radarseg/train/callbacks.py) and marked as "Custom>CallbackName".  
<br>
## FAQ
<details>
<summary>What does the value of the radius attribute represent?</summary>

The radius represents the aggregation area of the local neighborhoods and is given with respect to the spatial coordinates. If the data is normalized by a scaling function, the radius represents the used fraction of the normalization range.  
Example: If the x- and y-coordinates are scaled by a min value of -250m and a max value of 250m, a radius of 1 would correspond to a range of 500m.

</details>
<br />

<details>
<summary>How is the loss weight determined?</summary>

The loss weight is determined according to the [class weight script](radarseg/data/dataset_utils/get_class_weights.py).

</details>
<br />
