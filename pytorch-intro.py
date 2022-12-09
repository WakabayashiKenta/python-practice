import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#transformに前処理の定義、Composeは複数操作を簡単に記述できる。ToTensorでPILImageを値の範囲が[0,1]のテンソル型へ変換、Normalizeで標準化(タプルの左から平均、分散)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#引数のrootは保存先ディレクトリを指定、trainは訓練データかどうか、downloadはrootで指定したディレクトリにダウンロードするか、transformは定義した前処理を指定できる。
trainset = torchvision.datasets.CIFAR10(root='./data', train= True, download=True,transform=transform)

#訓練データの0番目のテンソルと正解ラベルを出力
print(trainset[0])

#DataLoaderでDatasetが使用できる。第一引数は利用するデータセット、バッチサイズは一回の訓練で何個データを使うか、shuffleは参照の仕方をランダムにするか、num_workerは複数処理をするかで2以上で値の数だけ並行処理をする。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2,)

#訓練データと同じようなやり方でテストデータを作る
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

print(trainset.data.shape)

print(testset.data.shape)

print(trainset.classes)

#何の画像かの種類
classes = trainset.classes

def imshow(img):
    #非正規化
    img = img / 2 + 0.5
    print(type(img))
    #torch.Tensor型からnumpy.ndarray型に変換
    npimg = img.numpy()
    print(type(npimg))

    print(npimg.shape)
    #軸の順番を入れ替える(RGB,縦,横)から(縦.横,RGB)
    npimg = np.transpose(npimg, (1,2,0))
    print(npimg.shape)

    plt.imshow(npimg)
    plt.show()

#iterは次の要素にアクセスすることを繰り返すインターフェースのこと、next()でどんどん取っていける.make_gridは複数の画像をグリッド状に並べた画像を作成できる
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

#ネットワークを作成
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #処理内容を定義
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#loss関数の定義
criterion = nn.CrossEntropyLoss()

#最適化関数の定義
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, start=0):
        inputs, labels = data    #17行目と同じようにテンソルと正解ラベルがdataには入っているのでそれぞれに代入
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
torch.save(net.state_dict(), PATH)    #学習したモデルの保存


#モデルを使う
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))    #各層のパラメータをtensorにアップする
outputs= net(images)
_, predicted = torch.max(outputs, 1)    #tensor配列の最大値の取得(第二引数はaxisで0ならcol)
print('Predicted:', ''.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0

with torch.no_grad():     #withをつかうことで計算グラフが作られず、計算グラフが作られない
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the NEtwork on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze() #次元削減
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))    #画像ごとの正答率を出力