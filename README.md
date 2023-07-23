# CONVOLUTION-NEURAL-NETWORK
A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data. This repository contains the code for CNN with a categorical classification dataset.

The three types of layers usually present in a Convolutional Network are:

Convolutional Layers 
Pooling Layers 
Fully Connected Layers 

The Convolution Layer:
Convolution is an orderly procedure where two sources of information are intertwined; it’s an operation that changes a function into something else. Convolutions have been used for a long time typically in image processing to blur and sharpen images, but also to perform other operations.

The Pooling Layer:
Pooling layers are used to reduce the dimensions of the feature maps. Thus, it reduces the number of parameters to learn and the amount of computation performed in the network.

The Fully Connected Layer:
Fully-connected layers, also known as linear layers, connect every input neuron to every output neuron and are commonly used in neural networks. If you want to know more about filly connected layer check my artificail neural network repositorie.

About the Dataset:
This is a categorical classification dataset. Which has a different categories of data. This dataset is from pytorch CIFAR-10. The CIFAR-10 dataset contains 60,000 32x32 color images, and each image is represented as a 3-D tensor. Each image will be a (32,32,3) tensor, where the dimensions are 32 (height) x 32 (weight) x 3 (R-G-B color channels). There are different types of classes are present. They are ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'].

This convolution neural network has Cross entropy loss function which is best suitable for the categorical dataset.
