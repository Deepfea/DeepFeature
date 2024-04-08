# DeepFeature: Guiding Deep Learning System Adversarial Testing using Robust Features

### Overview:

we propose an adversarial testing framework named DeepFeature, locating the robust features as the root cause. 
First, DeepFeature generates these features that are extracted when DNN decision-making. 
Next, DeepFeature locates the weak features under adversarial attacks, which fail to be transformed layer-by-layer and lead to the wrong outputs of DNNs. They are the main culprits of vulnerability. 
Finally, DeepFeature selects diverse samples containing weak features for adversarial retraining. 


### Setup:

We conduct the experiments on Tensorflow (v1.15.4) and Keras (v2.3.1). 
The physical host is a machine running on the Ubuntu 18.04 system, equipped with one Nvidia RTX 3090 GPU, the Intel i9-10900K(3.7GHz) CPU, and 64GB of RAM.

We utilize Anaconda 3 to manage all of the Python packages. To facilitate reproducibility 
of the Python environment, we release an Anaconda YAML specification file of the libraries 
utilized in the experiments. This allows the user to create a new virtual Python environment 
with all of the packages required to run the code by importing the YAML file. 

### Dataset:
We conduct the experiments on three datasets——MNIST,CIFAR-10 and ImageNet subset.

**MNIST:** It contains 60,000 images for training and 10,000 for testing. 
These images represent handwritten digits and belong to ten categories from 0 to 9. 
Each size is 28 * 28 * 1 pixels.

**CIFAR-10:** It contains 50,000 images for training and 10,000 for testing. 
Each image belongs to one of the ten categories, e.g., cats, dogs, trucks. 
Each size is 32 * 32 * 3 pixels. 

**ImageNet subset:** It, a real-world image dataset for recognizing objects, contains 50,000 images for training and 10,000 images for testing. 
Similar to CIFAR, these images belong to ten classes, e.g., birds, turtles, and Deers. 
Each size is 224 * 224 * 3 pixel.

### Models:
There are six models that we train and evaluate. 
We train the small-sized MNIST dataset on Lenet-1 and Lenet-5, respectively. 
We train the medium-sized CIFAR-10 dataset on ResNet-20 and VGG-16, respectively. 
We train the large-sized ImageNet dataset on ResNet-50 and VGG-19, respectively. 

### Robust Features:
The hidden layers implement complex and high-dimensional decision-making within the DNN. 
They extract and transform local features of the input data and determine the final output through nonlinear mapping. 
Layers closer to the output layer can extract more high-order features. 
That is high-order features, e.g., different parts of objects. 
High-order features more accurately express the relationship between input data and output results. 
They are more complex and more valuable to collect. 
High-order features are also known as robust features, which not only contain semantic information but also provide evidence to DNN decision-making directly.
In DeepFeature, we generate the robust features for understanding the internal logic of the model under adversarial attacks.

### Running the code:

Environment Setup: 
````
    1. Setup a Linux environment (not tested for Windows) with an Nvidia GPU containing at least 12GB of memory (less may work, but not tested).
    2. Download the open-sourced code, dataset and models.
    3. Create a virtual Python environment using the provided YAML configuration file on Github.
    4. Activate the new virtual Python environment
````
Parameters:

* Dataset(s): the user can select between a few datasets. 

* Model(s): the user can select between a few models. 

**Running DeepFeature:**
In order to run DeepFeature, run the file under approach. Parameter options refer to the paper.
