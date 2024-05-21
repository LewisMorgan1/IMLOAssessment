import os
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torch.optim as optim
import time
import argparse
import requests
import tarfile


# the application downloads the Oxford flower images:
# see https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html
# using pytorch torchvision.datasets.Flowers102
# However, the segmentation image masks that were linked to in the PDF
# specification 1.2 Dataset these must be downloaded separately and expanded
# into a segmim folder.
# see https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

class FlowersWithSegmentationDataset(Dataset):
    """
    A PyTorch Dataset class to apply the segmentation mask.

    Parameters:
        root_dir (str): Directory where the Flowers dataset is stored.
        mask_dir (str): Directory where segmentation masks are stored.
        split (str): One of 'train', 'val', or 'test' to show the dataset split.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.

   Raises:
       ValueError: If the `split` is not one of 'train', 'val', or 'test'.
   """

    def __init__(self, root_dir, mask_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load split information from setid.mat
        setid_path = os.path.join(root_dir, 'setid.mat')
        setid = scipy.io.loadmat(setid_path)

        # The id index is adjusted to start from 0.
        if split == 'train':
            split_indices = setid['trnid'][0] - 1
        elif split == 'val':
            split_indices = setid['valid'][0] - 1
        elif split == 'test':
            split_indices = setid['tstid'][0] - 1
        else:
            raise ValueError("Invalid split. Supported splits: 'train', 'val', 'test'")

        # Creates an array of all the image filenames
        self.image_files = [f'image_{str(idx + 1).zfill(5)}.jpg' for idx in split_indices]

        self.mask_directory = mask_dir
        # Creates an array of all the mask filenames
        self.mask_files = [f'segmim_{str(idx + 1).zfill(5)}.jpg' for idx in split_indices]

        # Creating an array of the labels for the mask
        imagelabels_path = os.path.join(root_dir, 'imagelabels.mat')
        imagelabels = scipy.io.loadmat(imagelabels_path)
        self.labels = imagelabels['labels'][0]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
          Fetches the image and mask at the index `idx`, applies the mask, and transforms the image if a transform is provided.

          Parameters:
              idx (int): Index of the image and mask to load.

          Returns:
              tuple: (transformed_image, label) where transformed_image is the tensor of the masked and transformed image,
                     and label is the corresponding label of the flower.
          """

        # Loads the image from file.
        img_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, 'jpg', img_name)
        image = Image.open(image_path).convert('RGB')

        # Displays the image
        # plt.imshow(image)
        # plt.show()

        # Loads the mask from the file
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_directory, mask_name)
        mask = Image.open(mask_path).convert('RGB')

        # Displays the mask image
        # plt.imshow(mask)
        # plt.show()

        # Apply mask to image
        mask = F2.to_tensor(mask).float()
        mask = (mask > 0).float()
        image = F2.to_tensor(image).float()
        masked_image = image * mask

        # Display the newly masked image
        # image2 = F2.to_pil_image(masked_image)
        # plt.imshow(image2)
        # plt.show()

        # the images start with names like image_00001.jpg but the array starts at zero
        label = self.labels[idx - 1]

        # If a transform is passed in, apply that transformation
        if self.transform:
            image = F2.to_pil_image(masked_image)
            image = self.transform(image)

        return image, label


# class to manage the loading, training, validation and testing of the flower CNN
class FlowerNN(nn.Module):
    """
       A PyTorch neural network module designed for classifying flower images, with configurable convolutional
        layers and other hyperparameter.

       Parameters:
           in_channels (int): Number of input channels.
           out_channels (list): List of output channels for each convolutional layer.
           flower_width (int): Width of the input images.
           flower_height (int): Height of the input images.
           flower_classes (int): Number of flower output classes.
           kernel_size (tuple): Size of the kernels.
           stride (tuple): Stride of the convolutions.
           padding (tuple): Padding added to both sides of the input.
           patience (int): Number of epochs with no improvement after which training will be stopped.
           lr (float): Learning rate of the NN.
           layer_count (int): Number of layers in the network.
       """

    def __init__(self, in_channels, out_channels, flower_width, flower_height, flower_classes, kernel_size, stride,
                 padding, patience, lr, layer_count):
        super(FlowerNN, self).__init__()

        index = 0
        divisor = 2
        pool_kernel_size = (2, 2)
        pool_stride = (2, 2)

        # Layer 1: Max pool, convolutional and batch normal. Construction of attributes.
        self.layers = layer_count
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[index], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels[index])

        if layer_count > 1:

            # Layer 2: Max pool, convolutional and batch normal. Construction of attributes.
            self.conv2 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm2 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 2:
            # Layer 3: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv3 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm3 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 3:
            # Layer 4: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv4 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool4 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm4 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 4:
            # Layer 5: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv5 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool5 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm5 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 5:
            # Layer 6: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv6 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool6 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm6 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 6:
            # Layer 7: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv7 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool7 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm7 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        if layer_count > 7:
            # Layer 8: Max pool, convolutional and batch normal. Construction of attributes.

            self.conv8 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            self.pool8 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.norm8 = nn.BatchNorm2d(out_channels[index + 1])
            divisor *= 2
            index += 1

        # 3 fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                in_features=int(flower_width / divisor) * int(flower_height / divisor) * out_channels[layer_count - 1],
                out_features=4096),
            # nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, flower_classes))

        self.dropout1 = nn.Dropout(p=0.5)

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Construct optimiser
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.SGD(self.parameters(), lr= LR, momentum=0.9)

        # Construct loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
       Applies each layer to the batch of images to get the predictions

       Parameter:
           x (Tensor): Input data tensor containing image data.

       Return:
           x: Output tensor with the network's predictions for each class.
       """

        # Applies a configurable number of layers
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        if self.layers > 1:
            x = self.pool2(F.relu(self.norm2(self.conv2(x))))

        if self.layers > 2:
            x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        if self.layers > 3:
            x = self.pool4(F.relu(self.norm4(self.conv4(x))))

        if self.layers > 4:
            x = self.pool5(F.relu(self.norm5(self.conv5(x))))

        if self.layers > 5:
            x = self.pool6(F.relu(self.norm6(self.conv6(x))))

        if self.layers > 6:
            x = self.pool7(F.relu(self.norm7(self.conv7(x))))

        if self.layers > 7:
            x = self.pool8(F.relu(self.norm8(self.conv8(x))))

        # Apply dropout
        x = F.relu(self.dropout1(x))

        # Reshape the data
        x = x.reshape(x.shape[0], -1)

        #  Apply the fully connected layers
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def show_lovely_bunch_of_flowers(self, image, labels):
        """
        Displays the images and their layers

        Parameters:
        image: batch of images
        labels: array of labels

        """
        print(f"labels={labels}")
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def download_and_extract(self, url, folder):
        """
        Downloading the segmentation tar file if it does not already exist locally

        Parameters:
            url: url to the tar file
            folder: the folder it will be saved in

        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        tar_filename = url.split('/')[-1]
        tar_path = os.path.join(folder, tar_filename)

        # if the tar file has not already been downloaded
        if not os.path.exists(tar_path):
            # Download tar file
            self.progress_bar(100, 0, barprefix="Downloading segmentation masks:")
            response = requests.get(url, stream=True)
            with open(tar_path, 'wb') as tar_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        tar_file.write(chunk)

            # Expand the tar file
            self.progress_bar(100, 50, barprefix="Downloading segmentation masks:")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=folder)

            self.progress_bar(100, 100, barprefix="Downloading segmentation masks:")

    def load_data_sets(self, num_workers, batch_size, image_width, image_height, mean, std_dev, sharp, contrast, solar,
                       equal, rotation):
        """
        Downloads the test data sets and defines the transforms.

        Parameters:
            num_workers: number of threads. 0 means single threaded
            batch_size: number of images in a batch
            image_width: image width in pixels
            image_height: image height in pixels
            mean: normalised mean
            std_dev: normalised standard deviation
            sharp: whether to apply random sharp transform
            contrast: whether to apply random contrast transform
            solar: whether to apply random solarised transform
            equal: whether to apply random equalised transform
            rotation: amount of random rotation in degrees

        """

        # Dynamically building an array of transforms
        transform_list = [transforms.CenterCrop(500), transforms.Resize((image_width, image_height)),
                          transforms.RandomVerticalFlip(),
                          transforms.RandomHorizontalFlip(), transforms.RandomRotation(rotation)]

        # based on the parameters, chose the transformation.
        if sharp:
            transform_list.append(transforms.RandomAdjustSharpness(sharpness_factor=2.0))
        if contrast:
            transform_list.append(transforms.RandomAutocontrast())
        if solar:
            transform_list.append(transforms.RandomSolarize(threshold=0.5))
        if equal:
            transform_list.append(transforms.RandomEqualize())

        # Add random affine transformation
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        # Normalise it
        transform_list.append(transforms.Normalize(mean, std_dev))

        train_transform = transforms.Compose(transform_list)

        # test simple transforms
        test_transform = transforms.Compose([
            transforms.CenterCrop(500),
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev)
        ])

        # download the images
        torchvision.datasets.Flowers102(root='./flowerdata', split="train", download=True)
        torchvision.datasets.Flowers102(root='./flowerdata', split="val", download=True)
        torchvision.datasets.Flowers102(root='./flowerdata', split="test", download=True)

        # download and apply the segmentation masks from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
        mask_dir = "segmim"
        self.download_and_extract("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz", ".")

        # apply the segmentation masks from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
        trainset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir=mask_dir, split="train",
                                                  transform=train_transform)
        valset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir=mask_dir, split="val",
                                                transform=train_transform)
        testset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir=mask_dir, split="val",
                                                 transform=test_transform)

        # Constructing loader attributes
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers)

        # show the images
        # images, labels = next(iter(self.test_loader))
        # self.show_lovely_bunch_of_flowers(torchvision.utils.make_grid(images),labels)

    def train_validate_test_network(self, num_epochs, batch_size, layers):
        """
        It trains and validates for a number of epochs. Every 50 epoch, it tests.
        Parameters:
            num_epochs: Number of training loops
            batch_size: Number of images in a single batcg
            layers: Number of dynamic layers.

        """

        accuracy_checked = False
        # Main training loop
        for epoch in range(num_epochs):
            accuracy_checked = False
            running_loss = 0
            num_of_training_images = len(self.train_loader)
            # Loop over training data
            for index, (img_batch, flower_type) in enumerate(self.train_loader):
                img_batch = img_batch.to(device)
                flower_type = flower_type.to(device)

                # given a batch of images, get the predictions
                out = self(img_batch)
                # calculate loss from the loss function
                loss = self.criterion(out, flower_type)
                # keeps track of loss
                running_loss += loss

                self.optimizer.zero_grad()
                # Backwards pass
                loss.backward()
                self.optimizer.step()
            # Visualise progress
            self.progress_bar(num_of_training_images, num_of_training_images,
                              barprefix=f"Epoch {epoch + 1:03d}/{num_epochs:03d}:")

            # Loop over the 'validation' data
            num_corrects = 0
            num_samples = 0
            with torch.no_grad():
                for index, (img_batch, flower_type) in enumerate(self.val_loader):
                    img_batch = img_batch.to(device)
                    flower_type = flower_type.to(device)

                    # Given a batch of validation images, find the predictions
                    out = self(img_batch)

                    # calculate the accuracy compared to known labels
                    _, predictions = out.max(1)
                    num_corrects += (predictions == flower_type).sum()
                    num_samples += predictions.size(0)

            print(
                f" Validation accuracy {num_corrects}/{num_samples}: {num_corrects / num_samples * 100:05.2f}% Loss: {running_loss:.{2}f} Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            # every 50th epoch  how accurate is the network
            if (epoch + 1) % 50 == 0:
                accuracy = self.test_network(batch_size, num_epochs)
                accuracy_checked = True
                # Checkpointing the training data

                # Periodically saving your trained model, so that if anything happened to the training run
                # (i.e. power cut, crashing, server reboots, etc), you can resume the training process by
                # loading the latest version of your trained model
                self.save_flower_training(layers=layers)
                # If it is getting worse after n times, stop.
                if self.early_stopping(accuracy):
                    print("Early stopping now")
                    break

        if not accuracy_checked:
            accuracy = self.test_network(batch_size, num_epochs)
            if self.early_stopping(accuracy):
                print("Early stopping now")

        print(f"Best accuracy {self.best_score:.{2}f}% {info}", flush=True)

    def early_stopping(self, score):
        """
        Keeps track of the best score so if after n times it is getting worse, stop.
        Parameter:
            score: Latest score

        Return:
            True is stop
            False is keep going
        """
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Let's stop now as we aren't really adding value, the best core is {self.best_score}%")
        return self.early_stop

    def test_network(self, batch_size, num_epochs):
        """
        Puts the network into test mode. It loops over the test images and get predictions and compares them against
         known labels to find the accuracy.

        Parameters:

            batch_size: Number of images in a batch
            num_epochs: Number of training loops

        Returns:
            Accuracy percentage
        """
        print("\n")
        num_corrects = 0
        num_samples = 0
        self.eval()  # put into testing mode

        with torch.no_grad():
            num_of_images = len(self.test_loader)
            for index, (x, y) in enumerate(self.test_loader):

                # sending the data to the device
                x = x.to(device)
                y = y.to(device)

                # forward
                y_hat = self(x)

                # calculate the accuracy by comparing predictions against known labels
                _, predictions = y_hat.max(1)
                num_corrects += (predictions == y).sum()
                num_samples += predictions.size(0)

                if (index) % 10 == 0:
                    self.progress_bar(num_of_images, index, barprefix="Test:")

            self.progress_bar(num_of_images, num_of_images, barprefix="Test:")

        print(
            f" Test accuracy: {num_corrects}/{num_samples}: {num_corrects / num_samples * 100:05.2f}% with best so far of {self.best_score}%")
        print("\n")
        self.train()  # put back into training mode

        return num_corrects / num_samples * 100

    def load_flower_training(self, layers):
        """
        If there is a trained model file, load it.
        Parameter:
            layers: Number of layers used in the file name
        """
        file_name = f"./flowers-trained-data-layers{layers}.pth"
        if os.path.exists(file_name):
            self.progress_bar(100, 0, barprefix="Loading training data:")
            self.load_state_dict(torch.load(file_name))
            self.progress_bar(100, 100, barprefix="Loading training data:")
        else:
            print(f"File {file_name} not found")

    def save_flower_training(self, layers):
        """
        Saves the trained model.

        Parameter:
            layers: Number of layers used in the file name
        """
        self.progress_bar(100, 0, barprefix="Saving training data:")
        torch.save(self.state_dict(), f"./flowers-trained-data-layers{layers}.pth")
        self.progress_bar(100, 100, barprefix="Saving training data:")

    def progress_bar(self, total, current, barprefix="", decimals=1, length=80, fill='█'):
        """
        Simple horizontal progress bar

        Parameters:
            total: Maximum value
            current: Current value
            barprefix: Optional text before the progress bar
            decimals: Number of decimal places
            length: How much of the screen to use
            fill: Character to fill
        """
        percent = ("{0:.{1}f}".format(100 * (current / float(total)), decimals)) + "%"
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f"\r{barprefix} |{bar}| {percent}", end="")  # Remove newline character
        print("\033[K", end="")  # escape char moved cursor to start of line


# Device works out the hardware available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")

# Command line arguments
parser = argparse.ArgumentParser(description="Flower classification")
parser.add_argument("--num_epochs", type=int, required=True,
                    help="the number of training runs to perform")
parser.add_argument("--batch_size", type=int, required=False, default=256,
                    help="the number of images in a batch")
parser.add_argument("--num_workers", type=int, required=False, default=0,
                    help="the number of background threads to use")
parser.add_argument("--patience", type=int, required=False, default=10,
                    help="how many epochs that don't add value before giving up early")
parser.add_argument("--image_width", type=int, required=False, default=500,
                    help="image width")
parser.add_argument("--image_height", type=int, required=False, default=500,
                    help="image height")
parser.add_argument("--load_training", type=bool, required=False, default=False,
                    help="load the old training data")
parser.add_argument("--sharp", type=bool, required=False, default=False)
parser.add_argument("--contrast", type=bool, required=False, default=False)
parser.add_argument("--solar", type=bool, required=False, default=False)
parser.add_argument("--equal", type=bool, required=False, default=False)
parser.add_argument("--rotation", type=int, required=False, default=70)
parser.add_argument("--layers", type=int, required=False, default=8)

args = parser.parse_args()

# parameters

arg_flower_classes = 102
arg_kernel_size = (3, 3)
arg_stride = (1, 1)
arg_padding = (1, 1)
arg_mean = (0.3451, 0.3045, 0.2363)  # from calcMeanAndStdDev.py
arg_std_dev = (0.3134, 0.2661, 0.2696)  # from calcMeanAndStdDev.py
arg_learning_rate = 0.001

arg_image_width = args.image_width  # default 200
arg_image_height = args.image_height  # default 200
arg_batch_size = args.batch_size  # default = 64
arg_num_workers = args.num_workers  # 0 = this makes it run as on single CPU/GPU
arg_num_epochs = args.num_epochs
arg_patience = args.patience  # default 10
arg_layers = args.layers  # default 7

info = f", layers={arg_layers}, num_epochs={arg_num_epochs}, batch_size={arg_batch_size}, image_width={arg_image_width}, image_height={arg_image_height}, rotation={args.rotation}, sharp={args.sharp}, contrast={args.contrast}, solar={args.solar}, equal={args.equal}"

start_time = time.time()

# Creating the CNN
flowerNN = FlowerNN(in_channels=3, out_channels=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                    flower_width=arg_image_width, flower_height=arg_image_height,
                    flower_classes=arg_flower_classes, kernel_size=arg_kernel_size, stride=arg_stride,
                    padding=arg_padding,
                    patience=arg_patience, lr=arg_learning_rate, layer_count=arg_layers)
# Use the appropriate hardware
flowerNN = flowerNN.to(device)
# Loads the data sets and transforms
flowerNN.load_data_sets(num_workers=arg_num_workers, batch_size=arg_batch_size, image_width=arg_image_width,
                        image_height=arg_image_height,
                        mean=arg_mean, std_dev=arg_std_dev,
                        sharp=args.sharp, contrast=args.contrast, solar=args.solar, equal=args.equal,
                        rotation=args.rotation
                        )

# Decide whether to preload or start from scratch
if args.load_training:
    flowerNN.load_flower_training(layers=arg_layers)

# Trains, validates and tests
flowerNN.train_validate_test_network(num_epochs=arg_num_epochs, batch_size=arg_batch_size, layers=arg_layers)
# Saves the trained model
flowerNN.save_flower_training(layers=arg_layers)

# Report duration
end_time = time.time()
execution_time_in_mins = (end_time - start_time) / 60
print(f"Duration {execution_time_in_mins:.{0}f} mins")
