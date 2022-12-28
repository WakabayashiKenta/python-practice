import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torchvision import models
from typing import List

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using devive: {device}')

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train= True, download=True,transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2,)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes: List[str] = trainset.classes

    dataiter = iter(trainloader)
    image, labels = next(dataiter)

    model_ft = models.resnet18(pretrained=False)
    num_ftrs: int = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))
    net = model_ft
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()    #勾配の初期化
            outputs = net(inputs)
            loss = criterion(outputs, labels)    #outputとlabelsのlossを計算

            loss.backward()    #lossから勾配情報の計算
            optimizer.step()    #勾配情報を使用してパラメータの更新
            train_loss = loss.item()    #テンソルの値を取得
            running_loss += loss.item()
            
            if i % 2000 ==1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    PATH = '.cifar_net.pth'
    torch.save(net.state_dict(), PATH) 

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images = images.to(device)
    labels = labels.to(device)

    outputs: torch.Tensor = net(images)
    _, predicted = torch.max(outputs, 1)

    correct: float = 0
    total: float = 0
    type(labels[1])

    with torch.no_grad():     #withをつかうことで計算グラフが作られず、計算グラフが作られない
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the NEtwork on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct: List[float] = list(0. for i in range(10))
    class_total: List[float] = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs: torch.Tensor = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze() #次元削減
            for i in range(4):
                label: torch.Tensor = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))    #画像ごとの正答率を出力
        """
        Resnet(Pretrain=False)
        Accuracy of airplane : 72 %
        Accuracy of automobile : 88 %
        Accuracy of  bird : 66 %
        Accuracy of   cat : 77 %
        Accuracy of  deer : 35 %
        Accuracy of   dog : 34 %
        Accuracy of  frog : 58 %
        Accuracy of horse : 73 %
        Accuracy of  ship : 70 %
        Accuracy of truck : 75 %
        """

        """
        net
        Accuracy of airplane : 56 %
        Accuracy of automobile : 85 %
        Accuracy of  bird : 34 %
        Accuracy of   cat : 49 %
        Accuracy of  deer : 45 %
        Accuracy of   dog : 36 %
        Accuracy of  frog : 61 %
        Accuracy of horse : 66 %
        Accuracy of  ship : 69 %
        Accuracy of truck : 55 %"
        """



