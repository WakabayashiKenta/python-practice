import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train= True, download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2,)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

print(trainset.data.shape)

print(testset.data.shape)

print(trainset.classes)

classes = trainset.classes

def imshow(img):


    img = img / 2 + 0.5
    print(type(img))
    npimg = img.numpy()
    print(type(npimg))

    print(npimg.shape)
    npimg = np.transpose(npimg, (1,2,0))
    print(npimg.shape)

    plt.imshow(npimg)
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

