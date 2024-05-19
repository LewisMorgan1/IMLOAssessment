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


# the application downloads the Oxford flower images:
# see https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html
# using pytorch torchvision.datasets.Flowers102
# However, the segmentation image masks that were linked to in the PDF
# specification 1.2 Dataset these must be downloaded separately and expanded
# into a segmim folder.
# see https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
class FlowersWithSegmentationDataset(Dataset):
    def __init__(self, root_dir, mask_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load split information from setid.mat
        setid_path = os.path.join(root_dir, 'setid.mat')
        setid = scipy.io.loadmat(setid_path)

        if split == 'train':
            split_indices = setid['trnid'][0] - 1
        elif split == 'val':
            split_indices = setid['valid'][0] - 1
        elif split == 'test':
            split_indices = setid['tstid'][0] - 1
        else:
            raise ValueError("Invalid split. Supported splits: 'train', 'val', 'test'")

        self.image_files = [f'image_{str(idx + 1).zfill(5)}.jpg' for idx in split_indices]

        self.mask_directory = mask_dir
        self.mask_files = [f'segmim_{str(idx + 1).zfill(5)}.jpg' for idx in split_indices]

        imagelabels_path = os.path.join(root_dir, 'imagelabels.mat')
        imagelabels = scipy.io.loadmat(imagelabels_path)
        self.labels = imagelabels['labels'][0]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, 'jpg', img_name)
        image = Image.open(image_path).convert('RGB')

        # plt.imshow(image)
        # plt.show()

        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_directory, mask_name)
        mask = Image.open(mask_path).convert('RGB')

        # plt.imshow(mask)
        # plt.show()

        mask = F2.to_tensor(mask).float()
        mask = (mask > 0).float()

        # Apply mask to image
        image = F2.to_tensor(image).float()
        masked_image = image * mask

        # image2 = F2.to_pil_image(masked_image)
        # plt.imshow(image2)
        # plt.show()

        # the images start with names like image_00001.jpg but the array starts at zero
        label = self.labels[idx - 1]

        if self.transform:
            image = F2.to_pil_image(masked_image)
            image = self.transform(image)

        return image, label


# class to manage the loading, training, validation and testing of the flower CNN
class FlowerNN(nn.Module):
    def __init__(self, in_channels, out_channels, flower_width, flower_height, flower_classes, kernel_size, stride,
                 padding, patience, lr, layer_count):
        super(FlowerNN, self).__init__()

        index = 0
        divisor = 2

        self.layers = layer_count
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[index], kernel_size=kernel_size,
                               stride=stride, padding=padding)

        if layer_count > 1:
            self.conv2 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 2:
            self.conv3 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 3:
            self.conv4 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 4:
            self.conv5 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 5:
            self.conv6 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 6:
            self.conv7 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 7:
            self.conv8 = nn.Conv2d(in_channels=out_channels[index], out_channels=out_channels[index + 1],
                                   kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 8:
            self.conv9 = nn.Conv2d(in_channels=out_channels[7], out_channels=out_channels[8], kernel_size=kernel_size,
                                   stride=stride, padding=padding)
            divisor *= 2
            index += 1

        if layer_count > 9:
            self.conv10 = nn.Conv2d(in_channels=out_channels[8], out_channels=out_channels[9], kernel_size=kernel_size,
                                    stride=stride, padding=padding)
            divisor *= 2
            index += 1

        self.fc = nn.Linear(
            in_features=int(flower_width / divisor) * int(flower_height / divisor) * out_channels[layer_count - 1],
            out_features=flower_classes)

        # self.fc_layers = nn.Sequential(
        #    nn.Linear(in_features=num_out_ch[3] * num_out_ch[3], out_features=512),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(p=0.5),
        #    nn.Linear(in_features=512, out_features=num_classes)
        # )

        self.dropout1 = nn.Dropout(p=0.5)

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.SGD(self.parameters(), lr= LR, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        if self.layers > 1:
            x = self.pool(F.relu(self.conv2(x)))

        if self.layers > 2:
            x = self.pool(F.relu(self.conv3(x)))

        if self.layers > 3:
            x = self.pool(F.relu(self.conv4(x)))

        if self.layers > 4:
            x = self.pool(F.relu(self.conv5(x)))

        if self.layers > 5:
            x = self.pool(F.relu(self.conv6(x)))

        if self.layers > 6:
            x = self.pool(F.relu(self.conv7(x)))

        if self.layers > 7:
            x = self.pool(F.relu(self.conv8(x)))

        if self.layers > 8:
            x = self.pool(F.relu(self.conv9(x)))

        if self.layers > 9:
            x = self.pool(F.relu(self.conv10(x)))

        x = F.relu(self.dropout1(x))

        # Bilinear pooling
        # _, c, h, w = x.size()
        # x = x.view(-1, c, h * w)
        # x = torch.bmm(x, x.transpose(1, 2)) / (h * w)  # Bilinear pooling
        # x = x.view(-1, c * c)

        x = self.fc(x.reshape(x.shape[0], -1))

        return x

    def showBatchOfFlowers(self, image, labels):
        print(f"labels={labels}")
        # image = image / 2 + 0.5
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def load_data_sets(self, num_workers, batch_size, image_width, image_height, mean, std_dev, sharp, contrast, solar,
                       equal, rotation):
        transform_list = [transforms.CenterCrop(500), transforms.Resize((image_width, image_height)),
                          transforms.RandomVerticalFlip(),
                          transforms.RandomHorizontalFlip(), transforms.RandomRotation(rotation)]

        if sharp:
            transform_list.append(transforms.RandomAdjustSharpness(sharpness_factor=2.0))
        if contrast:
            transform_list.append(transforms.RandomAutocontrast())
        if solar:
            transform_list.append(transforms.RandomSolarize(threshold=0.5))
        if equal:
            transform_list.append(transforms.RandomEqualize())

        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std_dev))

        train_transform = transforms.Compose(transform_list)

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

        # apply the segmentation masks from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
        trainset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir='segmim', split="train",
                                                  transform=train_transform)
        valset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir='segmim', split="val",
                                                transform=train_transform)
        testset = FlowersWithSegmentationDataset(root_dir='./flowerdata/flowers-102', mask_dir='segmim', split="val",
                                                 transform=test_transform)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers)

        # images, labels = next(iter(self.test_loader))
        # self.showBatchOfFlowers(torchvision.utils.make_grid(images),labels)

    def train_validate_test_network(self, num_epochs, batch_size, info):

        accuracy_checked = False
        for epoch in range(num_epochs):
            accuracy_checked = False
            running_loss = 0
            num_of_training_images = len(self.train_loader)
            for index, (img_batch, flower_type) in enumerate(self.train_loader):
                img_batch = img_batch.to(device)
                flower_type = flower_type.to(device)

                out = flowerNN(img_batch)
                loss = self.criterion(out, flower_type)
                running_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.progress_bar(num_of_training_images, num_of_training_images,
                              barprefix=f"Epoch {epoch + 1:03d}/{num_epochs:03d}:")

            # load the 'validation' data
            num_corrects = 0
            num_samples = 0
            with torch.no_grad():
                for index, (img_batch, flower_type) in enumerate(self.val_loader):
                    img_batch = img_batch.to(device)
                    flower_type = flower_type.to(device)

                    out = flowerNN(img_batch)

                    # calculate the accuracy
                    _, predictions = out.max(1)
                    num_corrects += (predictions == flower_type).sum()
                    num_samples += predictions.size(0)

            print(
                f" Validation accuracy {num_corrects}/{num_samples}: {num_corrects / num_samples * 100:05.2f}% Loss: {running_loss:.{2}f} Learning Rate: {self.optimizer.param_groups[0]['lr']}")

            # every 10th epoch lets see how accurate the network is
            if (epoch + 1) % 10 == 0:
                accuracy = self.test_network(batch_size, num_epochs)
                accuracy_checked = True
                if self.early_stopping(accuracy):
                    print("Early stopping now")
                    break

        if not accuracy_checked:
            accuracy = self.test_network(batch_size, num_epochs)
            if self.early_stopping(accuracy):
                print("Early stopping now")

        print(f"Best accuracy {self.best_score:.{2}f}% {info}")

    def early_stopping(self, score):
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

                # calculate the accuracy
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

    def progress_bar(self, total, current, barprefix="", decimals=1, length=80, fill='█'):
        percent = ("{0:.{1}f}".format(100 * (current / float(total)), decimals)) + "%"
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f"\r{barprefix} |{bar}| {percent}", end="")  # Remove newline character
        print("\033[K", end="")  # escape char moved cursor to start of line


# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")

parser = argparse.ArgumentParser(description="Flower classification")
parser.add_argument("--num_epochs", type=int, required=False, default=200,
                    help="the number of training runs to perform")
parser.add_argument("--batch_size", type=int, required=False, default=64,
                    help="the number of images in a batch")
parser.add_argument("--num_workers", type=int, required=False, default=0,
                    help="the number of background threads to use")
parser.add_argument("--patience", type=int, required=False, default=5,
                    help="how many epochs that don't add value before giving up early")
parser.add_argument("--image_width", type=int, required=False, default=256,
                    help="image width")
parser.add_argument("--image_height", type=int, required=False, default=256,
                    help="image height")

parser.add_argument("--sharp", type=bool, required=False, default=False)
parser.add_argument("--contrast", type=bool, required=False, default=False)
parser.add_argument("--solar", type=bool, required=False, default=False)
parser.add_argument("--equal", type=bool, required=False, default=False)
parser.add_argument("--rotation", type=int, required=False, default=30)
parser.add_argument("--layers", type=int, required=False, default=4)

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
arg_num_epochs = args.num_epochs  # default 200
arg_patience = args.patience  # default 5
arg_layers = args.layers

info = f", layers={arg_layers}, num_epochs={arg_num_epochs}, batch_size={arg_batch_size}, image_width={arg_image_width}, image_height={arg_image_height}, rotation={args.rotation}, sharp={args.sharp}, contrast={args.contrast}, solar={args.solar}, equal={args.equal}"

start_time = time.time()

flowerNN = FlowerNN(in_channels=3, out_channels=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                    flower_width=arg_image_width, flower_height=arg_image_height,
                    flower_classes=arg_flower_classes, kernel_size=arg_kernel_size, stride=arg_stride,
                    padding=arg_padding,
                    patience=arg_patience, lr=arg_learning_rate, layer_count=arg_layers)
flowerNN = flowerNN.to(device)
flowerNN.load_data_sets(num_workers=arg_num_workers, batch_size=arg_batch_size, image_width=arg_image_width,
                        image_height=arg_image_height,
                        mean=arg_mean, std_dev=arg_std_dev,
                        sharp=args.sharp, contrast=args.contrast, solar=args.solar, equal=args.equal,
                        rotation=args.rotation
                        )
flowerNN.train_validate_test_network(num_epochs=arg_num_epochs, batch_size=arg_batch_size, info=info)

end_time = time.time()
execution_time_in_mins = (end_time - start_time) / 60
print(f"Duration {execution_time_in_mins:.{0}f} mins")



