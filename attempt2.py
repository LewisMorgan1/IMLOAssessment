import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse

# see https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html

# Hyperparameters
PATIENCE = 15
NUM_EPOCHS = 300
NUM_OUT_CH = [8, 16, 32, 64]
IMAGE_W = 150
IMAGE_H = 150
BATCH_SIZE = 64
LR = 0.001
NUM_WORKERS = 0
MEAN = (0.3451, 0.3045, 0.2363)
STD_DEV = (0.3134, 0.2661, 0.2696)

class FlowerNN(nn.Module):
    def __init__(self, num_channels=3, num_out_ch=[8, 16], img_w=100, img_h=100, num_classes=102):
        super(FlowerNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_out_ch[0],
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=num_out_ch[0], out_channels=num_out_ch[1],
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=num_out_ch[1], out_channels=num_out_ch[2],
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=num_out_ch[2], out_channels=num_out_ch[3],
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc = nn.Linear(in_features=int(img_w / 16) * int(img_h / 16) * num_out_ch[3], out_features=num_classes)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout with 50% probability

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.dropout1(x))

        x = self.fc(x.reshape(x.shape[0], -1))

        return x


"""Early stops the training if validation loss doesn't improve."""
class EarlyStopping:

    def __init__(self, patience=PATIENCE, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Let's stop now as we aren't really adding value, the best core is {self.best_score}")

def check_accuracy(loader, model):
  print("\n")
  num_corrects = 0
  num_samples = 0
  model.eval() # put into testing mode

  with torch.no_grad():
    num_of_images = len(loader)
    for index, (x,y) in enumerate(loader):

      # sending the data to the device
      x = x.to(device)
      y = y.to(device)

      # forward
      y_hat = model(x)

      # calculate the accuracy
      _, predictions = y_hat.max(1)
      num_corrects += (predictions == y).sum()
      num_samples += predictions.size(0)

      if (index) % 10 == 0:
         progress_bar(num_of_images, index, barprefix="Test:")

    progress_bar(num_of_images, num_of_images, barprefix="Test:")

  print(f"\n Test accuracy: {num_corrects}/{num_samples}: {num_corrects/num_samples*100:05.2f} % with {BATCH_SIZE} batch size and {NUM_EPOCHS} epochs")
  model.train() # put back into training mode

  return num_corrects/num_samples*100

start_time = time.time()

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")

# model
model = FlowerNN(num_channels=3, num_out_ch=NUM_OUT_CH, img_w=IMAGE_W, img_h=IMAGE_H, num_classes=102)
model = model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = LR)
#optimizer = optim.SGD(model.parameters(), lr= LR, momentum=0.9)

# Loss Function
criterion = nn.CrossEntropyLoss()

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD_DEV)
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD_DEV)
])

trainset = torchvision.datasets.Flowers102(root='./flowerdata5', split="train", download=True, transform=train_transform)
valset = torchvision.datasets.Flowers102(root='./flowerdata5', split="val", download=True, transform=train_transform)
testset = torchvision.datasets.Flowers102(root='./flowerdata5', split="test", download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def progress_bar( total, current, barprefix="", decimals=1, length=80, fill='█'):
    percent = ("{0:.{1}f}".format(100 * (current / float(total)), decimals)) + "%"
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f"\r{barprefix} |{bar}| {percent}", end="")  # Remove newline character
    # Move cursor to the beginning of the progress bar using escape sequences
    print("\033[K", end="")  # Clear the line from cursor to the end

def train_data(train_loader, val_load):

    # early stopping validation
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    best_accuracy = 0.0
    accuracy_checked = False
    for epoch in range(NUM_EPOCHS):
      accuracy_checked = False
      running_loss = 0
      num_of_training_images = len(train_loader)
      for index, (x,y) in enumerate(train_loader):
        # send the data to the device
        x = x.to(device)
        y = y.to(device)

        # forward
        y_hat = model(x)
        loss = criterion(y_hat, y)
        running_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

     #   if index % 10 == 0:
     #        progress_bar(num_of_training_images, index, barprefix=f"Epoch {epoch+1:03d}:")

      progress_bar(num_of_training_images, num_of_training_images, barprefix=f"Epoch {epoch+1:03d}:")

      # load the 'validation' data
      num_corrects = 0
      num_samples = 0
      with torch.no_grad():
        for index, (x,y) in enumerate(val_load):

          # sending the data to the device
          x = x.to(device)
          y = y.to(device)

          # forward
          y_hat = model(x)

          # calculate the accuracy
          _, predictions = y_hat.max(1)
          num_corrects += (predictions == y).sum()
          num_samples += predictions.size(0)

      print(f" Validation accuracy {num_corrects}/{num_samples}: {num_corrects / num_samples * 100:05.2f}% Loss: {running_loss:.{2}f}")

      early_stopping(running_loss, model)

      if early_stopping.early_stop:
          print("Early stopping triggered")
          break

      if (epoch+1) % 10 == 0:
          test_accuracy =  check_accuracy(test_loader, model)
          accuracy_checked = True
          if best_accuracy < test_accuracy:
              best_accuracy = test_accuracy

    if not accuracy_checked:
        test_accuracy = check_accuracy(test_loader, model)
        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy

    print(f"Best accuracy {best_accuracy}%")

train_data(train_loader,val_loader)

end_time = time.time()  # Record end time
execution_time_in_mins = (end_time - start_time) / 60
print(f"Duration {execution_time_in_mins:.{0}f} mins")

