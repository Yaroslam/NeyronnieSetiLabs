#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
from torch.utils.data import random_split
from net import VGG16_NET
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=64
num_epochs=5
lr=1e-4
class_size=10

tranform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#prep the train, validation and test dataset
torch.manual_seed(2021)
train = torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=tranform_train)
val_size = 10000
train_size = len(train) - val_size
train, val = random_split(train, [train_size, val_size])
test = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=tranform_test)

#  train, val and test datasets to the dataloader
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16_NET()
model = model.to(device=device)
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr)

for epoch in range(num_epochs):  # I decided to train the model for 50 epochs
    loss_var = 0

    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        print(idx)
        if idx % 64 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(train_loader)}] || Loss:{loss_var / len(train_loader)}')
    print(f"Loss at epoch {epoch + 1} || {loss_var / len(train_loader)}")

    with torch.no_grad():
        correct = 0
        samples = 0
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum()
            samples += preds.size(0)
        print(
            f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")

torch.save(model.state_dict(), "cifar10_vgg16_model.pt") #SAVES THE TRAINED MODEL
model = VGG16_NET()
model.load_state_dict(torch.load("cifar10_vgg16_model.pt")) #loads the trained model
model.eval()

test_loader = DataLoader(test, batch_size=8, shuffle=False)
correct = 0
samples = 0
for idx, (images, labels) in enumerate(test_loader):
    images = images.to(device='cpu')
    labels = labels.to(device='cpu')
    outputs = model(images)
    _, preds = outputs.max(1)
    correct += (preds == labels).sum()
    samples += preds.size(0)
print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")

