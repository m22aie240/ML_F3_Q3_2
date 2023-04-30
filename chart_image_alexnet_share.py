import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
#from google.colab import drive




from PIL import Image
# %matplotlib inline

#drive.mount('/content/grive/')

class ChartDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx, 0]) + ".png")
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1]

        # Convert label using the mapping dictionary and then to a torch tensor
        label = label_mapping[label]
        label = torch.tensor(label, dtype=torch.long)

        return image, label




# Read CSV files
train_df = pd.read_csv('./charts/train_val.csv')
train_df, val_df = train_test_split(train_df, test_size=0.2)
unique_labels = train_df['type'].unique()
label_mapping = {label: index for index, label in enumerate(unique_labels)}


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create datasets
train_dataset = ChartDataset(train_df, './charts/train_val', transform=transform)
val_dataset = ChartDataset(val_df, './charts/train_val', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


# Load the pre-trained AlexNet model
from torchvision.models.alexnet import AlexNet_Weights

model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)


# Modify the first layer to accept 4 input channels
model.features[0] = nn.Conv2d(4, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

# Replace the last layer to match the number of classes in your dataset (5 classes in your case)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 5)

# Rest of the code

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print('Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))

model.eval()
running_loss = 0.0
running_corrects = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

val_loss = running_loss / len(val_dataset)
val_acc = running_corrects.double() / len(val_dataset)

print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))