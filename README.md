# Classification of Flower Images Using a Convolutional Neural Network

By Lewis Morgan, Y3921704, lm2283@york.ac.uk


This project implements a Convolutional Neural Network (CNN) for classifying flower images from the 102 Oxford Flowers dataset using PyTorch.

## Features

* Leverages PyTorch for building and training the CNN model.
* Achieves high accuracy on the 102 Oxford Flowers dataset.
* Provides options to customize training parameters through arguments.
* 65% accuracy using 8 layers and the trained data I've created from scratch.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/LewisMorgan1/IMLOAssessment.git

    cd IMLOAssessment
    ```

2. **Install the required packages**:
    ```sh
    pip install torch torchvision scipy matplotlib argparse numpy requests
    ```

3. **Strongly recommend running on a computer with a fast GPU or NPU (Neural processing unit). The performance difference was notable compared to a CPU.**


## Usage

```sh
python attempt5.py --num_epochs=<num_epochs> [--other_params]
```

Parameters:

```--num_epochs```: Number of training epochs, every 50th epoch and at the end it will test and report accuracy.

```--layers ```: Number of convolution layers to use in the CNN (default 7).

```--batch_size ```: Number of images in each batch (default 64).

```--num_workers ```: Number of worker threads, 0 means single threaded (default 0).

```--patience ```: How many epochs where the testing is not improoved before giving up (default 10).

```--image_width --image_height```: Image dimentions in px (default 500)

```--sharp ```: True/False - randomly adjust image sharpeness (default False)

```--contrast ```: True/False - randomly auto-contrast the image (default False)

```--solar ```: True/False - andomly solarise the image (default False).

```--equal ```: True/False - randomly equalise the image (default False).

```--rotation ```: randomly rotate the image by the degrees (default 70).

```--load_training ```: load a training data file from the last run (default True).

The command downloads the Oxford 102 flower dataset and also the flower segmentation masks if they are not available locally. 

If it is the first time running then it will take a little longer to download.

### Using the trained model I've created

If you just want to test a small number using the best defaults and loading from trained data this this will suffice:
```sh
python attempt5.py --num_epochs=20 --load_training=True
```
### Training from scratch

I found this to be the most effective way of training from scratch to acheive the highest accurancy. Each iteration adds to the previous but with different batch sizes:

```sh
# start training, it will download the images, and might take a min to get going, trains with a large batch size of 256

python attempt5.py --num_epochs=300 --batch_size=256 --image_width=500 --image_height=500 --sharp=True --rotation=70 --equal=True --patience=30 --layers=8 

# now start again but loading the training data from last time, this time with a batch size of 128

python attempt5.py --num_epochs=300 --batch_size=128 --image_width=500 --image_height=500 --sharp=True --rotation=70 --equal=True --patience=30 --layers=8 --load_training=True

# as above but progressively smaller batch size

python attempt5.py --num_epochs=300 --batch_size=64 --image_width=500 --image_height=500 --sharp=True --rotation=70 --equal=True --patience=30 --layers=8 --load_training=True

# as above but progressively smaller batch size

python attempt5.py --num_epochs=300 --batch_size=32 --image_width=500 --image_height=500 --sharp=True --rotation=70 --equal=True --patience=30 --layers=8 --load_training=True

# I've acheived ~65% accuracy with this approach. 
# However, it takes about 9 hrs on my computer.
```

## Files

```calcMeanAndStdDev.py```: prints the mean and std dev of the image set.

```attempt1.py```: Initial implementation of the CNN with two layers.

```attempt2.py```: Added arguments to control the hyper-parmeters and better progress indication.

```attempt3.py```: Added configurable layers and download and apply segmentation masks.

```attempt4.py```: VGG-16 algorithm from scratch - but slow, memory hungry and not actually very accurate. I think for success it needs to run for much longer than 12 hours.

```attempt5.py```: Final version. 8 configurable layers. Able to tune through command line arguments. Applies segmentation masks. Each layer being; Conv2d, BatchNorm2d and MaxPool2d. Then dropout and three fully connected linear transformations. Is not too slow but using GPU or NPU is recommended. 

```flowers-trained-data-layers8.pth```: Saved training data for a 8 layer CNN - 65% accuracy

```training-results.xlsx```: Results in tabular form with a nice graph