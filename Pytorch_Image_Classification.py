import os
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy

from typing import Dict, List, Tuple
from nptyping import NDArray

import matplotlib.pyplot as plt

def make_filepath_list() -> Tuple[List[str],List[str]]:
    """
    学習データ、検証データをそれぞれのファイルへのパスを格納したリストを返す
    os関連の扱い方はhttps://www.sejuku.net/blog/67787    に分かりやすくかいている

    Returns
    -------
    train_file_list: list
        学習データファイルへのパスを格納したリスト
    valid_file_list: list
        検証データファイルへのパスを格納したリスト
    """
    train_file_list = []
    valid_file_list = []

    for top_dir in os.listdir('./Images/'):    #os.listdir()はファイルやディレクトリの一覧を確認できる
        file_dir: str = os.path.join('./Images/', top_dir)    #os.pathはパス関連の操作を行う。.joinはパスやファイル名を結合
        file_list: List[str] = os.listdir(file_dir)    #最初のlistdirでディレクトリを取得し、この行のlistdirでファイル名を取得している。
                                            #ファイルはwordとかexcelで作られる文章、画像とかのことでディレクトリはフォルダをまとめたもの、

        num_data: int = len(file_list)
        num_split: int = int(num_data * 0.8)

        train_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/')
for file in file_list[:num_split]]    #top_dirの中身はn02115913-dholeとかn02116738-African_hunting_dogみたいな感じでfileの中身はn02116738_9924.jpgこんな感じ
        valid_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/')
for file in file_list[num_split:]]

    return train_file_list, valid_file_list

#train_file_list, valid_file_listの中身は['./Images/n02113799-standard_poodle/n02113799_4454.jpg', './Images/n02113799-standard_poodle/n02113799_4458.jpg', './Images/n02113799-standard_poodle/n02113799_447.jpg', './Images/n02113799-standard_poodle/n02113799_448.jpg'・・・・]みたいな感じ

#__next__はイテレータとしてよばれるときにある処理をしてから返す。例えば__next__() num+=2 だとしたら、　for i in a: print(i)とするとiは　１，３，５，７みたいな感じで呼び出される 
class ImageTransform(object):
    """
    入力画像の前処理クラス
    画像のサイズをリサイズする
    
    Attributes
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize: float, mean: float, std: float):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                transforms.Resize((resize,resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class DogDataset(data.Dataset):
    """
    犬種のDatasetクラス
    PytorchのDatasetクラスを継承させる
    
    Attributes
    ----------
    file_list: list
       画像のファイルパスを格納したリスト
    classes: list
        犬種のラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証かを設定
    """

    def __init__(self, file_list: List[str], classes: List[str], transform=None, phase='train'):
        self.file_list = file_list
        self.classes = classes
        self.transform = transform
        self.phase = phase

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)

    def __getitem__(self, index):    #__getitem__はオブジェクトに各括弧でアクセスした時の挙動を定義できる
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        #指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)

    
        #画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        #画像ラベルをファイル名から抜き出す
        label: int = self.file_list[index].split('/')[2][10:]

        label: int = self.classes.index(label)

        return img_transformed, label


class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
padding=1)    #kernel_sizeは畳み込みのフィルターの大きさ strideはどれだけずらすか、paddingは画像の端にあるデータを使った畳み込み処理ができるようにするために画像データの上下左右に要素を加えること
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=128 * 75 * 75, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=5)

    
    def forward(self, x):
        #畳み込みーReLUーPoolingの作業を繰り返しネットワークを深くし、精度を上げている。
        x = F.relu(self.conv1_1(x))    #畳み込みをした後に活性化関数ReLUを用いることで勾配消失問題に対応
        x = F.relu(self.conv1_2(x))    #畳み込みをしたものに畳み込みをすることで一回目の畳み込みで特徴抽出したものをさらに小さく特徴を捉えることができるようになっている。
        x = self.pool1(x)    #畳み込み層で出てきた特徴をより強調して学習させることが目的。これにより同じものが移った別の画像でも特徴検知が可能になり、汎化能力向上。畳み込みで得た情報を集約。圧縮の効果は微弱な変化位置に対して敏感になること、ある程度の過学習の抑制が可能

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 128 * 75 * 75)    #第一引数は何行か(第一引数に-1をいれると第二引数の列数に自動的にあわせて行を作成してくれる)、第二引数は何列かを指定し、調整してくれる。平坦化している
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)    #入力データを0~1の確率値になおす.dim=1は

        return x


if __name__ == '__main__':

#make_filepath_listの部分  
    dog_classes: List[str] = [
    'Chihuahua', 'Shih-Tzu',
    'borzoi', 'Great_Dane', 'pug'
    ]

    train_file_list, valid_file_list = make_filepath_list()

    print('学習データ数 : ', len(train_file_list))

    print(train_file_list[:3])

    print('検証データ数 : ', len(valid_file_list))

    print(valid_file_list[:3])

    resize: int = 300

    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    train_dataset = DogDataset(
        file_list = train_file_list, classes=dog_classes,
        transform=ImageTransform(resize, mean, std),
        phase='train'
    )
    valid_dataset = DogDataset(
        file_list=valid_file_list, classes=dog_classes,
        transform=ImageTransform(resize, mean, std),
        phase='valid'
    )
    index: int = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])
    print(train_dataset.__len__())
    print(valid_dataset.__len__())

    batch_size: int = 64
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False)

    dataloaders_dict = {
        'train': train_dataloader,
        'valid': valid_dataloader
    }
    
    batch_iterator = iter(dataloaders_dict['train'])
    inputs, labels = next(batch_iterator)

    print(inputs.size())
    print(labels)

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    num_epochs: int = 30

    for epoch in range(num_epochs):
        print(f'Epoch{epoch+1}/{num_epochs}')
        print('-----------')

        for phase in ['train', 'val']:

            if phase == ['train', 'val']:
                net.train()
            else:
                net.eval()

            epoch_loss: float = 0.0

            epoch_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)

                    loss = criterion(outputs, 1)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                    
                    epoch_loss += loss.item() * inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels.data)


            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    x = torch.randn([1, 3, 300, 300])