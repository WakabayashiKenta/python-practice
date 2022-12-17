import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn  
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from nptyping import NDArray, Shape, Float, Int0
from typing import Union, List, Dict, Optional


def imshow(inp: Union[torch.Tensor,NDArray[Shape["Any,Any"],Float]], title: Union[List[str],Optional[str]]=None) -> None:
    
    inp = inp.numpy().transpose((1,2,0))    #Pytorchにおいては基本的にテンソルは[Channel, Height, Width]の三次元で考える
    mean: NDArray[Shape["3, 1"], Float] = np.array([0.485, 0.456, 0.406])
    std: NDArray[Shape["3, 1"], Float] = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)    #第一引数が処理する配列、第二引数が最小値、第三引数が最大値である。配列の要素を最小値と最大値の間に収まるようにする
    plt.imshow(inp)    #imshowはmatplotlib.pyplotの関数で画像を表示する。
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()    #time.time()は現在時刻を取得できる 1670915199.4533744

    #=演算子でのコピーは全く同一のオブジェクトIDで同じオブジェクトなのでコピーされた側の変更はコピーした側にも反映される。copy(浅いコピー)はappendなどでは反映されないが、共通して持っている要素への変更は反映される。深いコピーは全く別のオブジェクトを作る。
    best_model_wts = copy.deepcopy(model.state_dict())    #model.state_dict()はモデルのweightやbias、その他、lrやmomentumといった情報を出力する
    best_acc: float = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()    #モデルを訓練用にする。Batch NormalizationやDropoutといったネットワークの学習プロセスを全体的に安定化させて学習スピードを上げる手法を有効にする
            else:
                model.eval()    #モデルを検証用にする。Batch NormalizationとDropoutを無効にする(というのもBatch NormalizationやDropoutはあくまで学習時のテクニックであって検証には必要ないから)

            running_loss: float = 0.0
            running_corrects: Union[torch.Tensor,int] = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()    #勾配を0に初期化する

                with torch.set_grad_enabled(phase == 'train'):    #勾配計算をするかどうかのスイッチ(訓練時にはするという設定)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)    #Tensor配列の最大値を返す
                    loss = criterion(outputs, labels)    #batchlossの平均からepochlossの計算

                    if phase == 'train':
                        loss.backward()    #backwardはrequires_grad=Trueとした変数に対して目的の関数に対しての微分を行ったときの勾配の計算ができる。
                        optimizer.step()   #stepは学習率と最適化手法に基づいて重みを更新している

                running_loss += loss.item() * inputs.size(0)    #loss.item()は損失を返す。これはミニバッチ全体の損失が含まれているのでバッチサイズをかけてlossを修正する
                running_corrects += torch.sum(preds == labels.data)    #正解率を表す
            if phase == 'train':
                scheduler.step()                                #schedulerは途中で学習率を変えてくれる
            
            epoch_loss = running_loss /dataset_sizes[phase]    #そのepochにおける損失
            epoch_acc = running_corrects.double() / dataset_sizes[phase]    
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')    

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6) -> None:
    was_training = model.training
    model.eval()
    images_so_far: float = 0
    fig = plt.figure()

    with torch.no_grad():    #このコードのネストの中はrequires_grad=Falseとなって計算されなくなる
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)      


if __name__ == '__main__':

    cudnn.benchmark = True    #ネットワークの形が固定のとき、GPU側でネットワークの計算を最適化し高速にする(データの入力サイズが最初屋途中で変わらない場合はTrueでいい、デメリットは計算の再現性はなくなること)
    plt.ion()    #インタラクティブにグラフを作成できる

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir: str = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(data_dir,
                                            data_transforms[x])
                                            for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)    #DataloaderはDatasetからサンプルを取得してミニバッチを作成する。サンプルを取得するDatasetとバッチサイズを指定する。DataLoadeはiterateするとミニバッチを返す。
                    for x in ['train', 'val']}
    dataset_sizes: Dict[str,int] = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names: List[str] = image_datasets['val'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputs, classes = next(iter(dataloaders['train']))

    out: torch.Tensor = torchvision.utils.make_grid(inputs)    #複数の画像をグリッド状に並べた画像を作成できる。(ただし直接的に画像ではなく、返り値はテンソル)
    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet18(weights=True)    #事前に訓練されたWeightを使用する
    num_ftrs: int = model_ft.fc.in_features    #resnet18 の入力の数を取得している
    
    model_ft.fc = nn.Linear(num_ftrs, 2)    #入力がnum_ftr,出力が2のLinearモデルにする

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)    #optimizerを初期化している、最適化関数をSGD(確率的勾配降下法にして実装))

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    #step_sizeで学習率を下げる周期を指定し、gammaで一度に学習率をどれだけ減らすかを指定する
    
    #train and evaluate
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    visualize_model(model_ft)

    model_ft = models.resnet18(pretrained=True)    #resnet18 の最下層の出力次数を取得している
    num_ftrs; int = model_ft.fc.in_features    #出力サイズが２になるようにしている   
   
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_df = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()


    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    visualize_model(model_ft)

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_glad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler,num_epochs=25)

    
    visualize_model(model_conv)   

    plt.ioff()
    plt.show()

