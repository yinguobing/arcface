# arcface
A TensorFlow implementation of [ArcFace](https://arxiv.org/abs/1801.07698) for face recognition.

![demo](docs/demo.gif)

Video Demo: [TensorFlow Face Recognition Demo (Bilibili)](https://www.bilibili.com/video/BV16T4y1P72j/)

## Features
 - Build with TensorFlow 2.4 Keras API
 - Advanced model architecture: HRNet v2
 - Build in softmax pre-training
 - Fully customizable training loop from scratch
 - Automatically restore from previous checkpoint, even epoch not accomplished.
 - Handy utilities to convert and shuffle the training datasets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.4-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.3-brightgreen)
![Numpy](https://img.shields.io/badge/Numpy-v1.17-brightgreen)

### Installing
#### Get the source code for training

```bash
# From your favorite development directory
git clone --recursive https://github.com/yinguobing/arcface
```

#### Download the training data
You can use any dataset as long as they can be converted to TensorFlow Record files. If you do not have any dataset, please download one from the ArcFace official [dataset list](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). 

#### Generate training dataset
This project provides a demo file showing how to convert the downloaded MXNet dataset into TFRecord files. You can run it like this:

```bash
# From the project root directory
python3 -m utils.mx_2_tf
```

#### Fully shuffle the dataset
Most face recognition datasets contains millions of training samples. It is better to fully shuffle the data in the record file.
```bash
cd utils
python3 shard_n_shuffle.py
```
Do not forget setting a correct dataset file path.


## Training
Deep neural network training can be complicated as you have to make sure everything is ready like datasets, checkpoints, logs, etc. But do not worry. Following these steps you should be fine.

### Setup the model
In the module `train.py`, setup your model's name.

```python
# What is the model's name?
name = "hrnetv2"
```

### Setup the training dataset
These files do not change frequently so set them in the source code.

```python
# Where is the training file?
train_files = "/path/to/train.record"

# How many identities do you have in the training dataset?
num_ids = 85742

# How many examples do you have in the training dataset?
num_examples = 5822653
```

### Setup the model parameters
These values varies with your dataset. Better double check before training.
```python
# What is the shape of the input image?
input_shape = (112, 112, 3)

# What is the size of the embeddings that represent the faces?
embedding_size = 512
```

### Start training
Set the hyper parameters in the command line. For softmax pre-training set `--softmax=True`. Otherwise the ArcLoss will be used.

```bash
# Softmax pre-training
python3 train.py --epochs=2 --batch_size=192 --softmax=True

# Train with ArcLoss
python3 train.py --epochs=4 --batch_size=192
```

Training checkpoints can be found in directory `checkpoints`. There is also another directory `model_scout` containing the best(max accuracy) model checkpoint. You get this feature for free.

### Resume training
Once the training was interrupted, you can resume it with the exact same command used for staring. The build in `TrainingSupervisor` will handle this situation automatically, and load the previous training status from the latest checkpoint. 

```bash
# The same command used for starting training.
python3 train.py --epochs=4 --batch_size=192
```

However, if you want more customization, you can also manually override the training schedule with `--override=True` to set the global step, the epoch and the monitor value.

```bash
python3 train.py --epochs=4 --batch_size=192 --override=True
```

### Monitor the training process
Use TensorBoard. The log and profiling files are in directory `logs`

```bash
tensorboard --logdir /path/to/arcface/logs
```

## Export
Even though the model wights are saved in the checkpoint, it is better to save the entire model so you won't need the source code to restore it. This is useful for inference and model optimization later.

### For cloud/PC applications
Exported model will be saved in `saved_model` format in directory `exported`. You can restore the model with `Keras` directly.

```bash
python3 train.py --export_only=True
```

## Inference
Once the model is exported, you can use `predict.py` to recognize faces. Please prepare some sample images and set them in the file. Then run

```bash
python3 predict.py
```

The most similar sample pairs will be printed. You can also use threshold to filter the results.

## Authors
Yin Guobing (尹国冰) - [yinguobing](https://yinguobing.com)

![wechat](docs/wechat.png)

## License
![GitHub](https://img.shields.io/github/license/yinguobing/arcface)

## Acknowledgments
- [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition): the official implementation in MXNet.
- [InsightFace-tensorflow](https://github.com/luckycallor/InsightFace-tensorflow): Tensoflow implementation of InsightFace.
- [Keras_insightface](https://github.com/leondgarse/Keras_insightface): Insightface Keras implementation
