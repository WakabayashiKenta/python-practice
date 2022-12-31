import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import sys
sys.path.append('/vision/references/detection')
import transforms as T
from engine import train_one_epoch, evaluate
import utils
from typing import List, Tuple

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms=None) -> None:
        self.root = root
        self.transforms = transforms
        #画像の並び方を整えるために、すべての画像ファイルをロードしてソートする。
        self.imgs: list[str] = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))#listdirは名前の通り、指定したパスのディレクトリ以下の部分を全部取ってきてリストにする。この場合、root/PNGImages以下を取ってくるから、FudaPed00001.pngとかが取ってこられている。
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx: int):
        img_path: str = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path: str = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)

        mask = np.array(mask)

        obj_ids:list[int] = np.unique(mask)

        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs: int = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos:tuple[np.ndarray[int]] = np.where(masks[i])
            xmin: int = np.min(pos[1])
            xmax: int = np.max(pos[1])
            ymin: int = np.min(pos[0])
            ymax: int = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) #バウンディングボックスの座標
        labels: torch.Tensor = torch.ones((num_objs,), dtype=torch.int64) #各バウンディングボックスのラベル(背景のクラス、犬のクラス、猫のクラスなどが識別される、)、0は常に背景のクラスでモデルに背景とみなされる。なので背景クラスがないならラベルに0を使ってはいけない(犬:1,猫:2のような感じでラベルをつける)。
        masks = torch.as_tensor(masks, dtype=torch.uint8)    #各オブジェクトに対応するセグメンテーションマスク

        image_id: torch.Tensor = torch.tensor([idx]) #画像の識別子
        area: torch.Tensor = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) #バウンディングボックスの面積、COCOメトリクスの評価時にメトリクスのスコアを小。中、大に分けるために使用
        iscrowd:torch.Tensor = torch.zeros((num_objs,), dtype=torch.int64) #iscrowd=Trueのインスタンスは評価中に無視される

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
#Mask R CNNは画像のオブジェクトに対してバウンディングボックス(オブジェクトが写っている範囲)とクラススコア(そのオブジェクトが何か)の両方を予測するモデル。あるオブジェクトらしさを算出してソフトマックスで

"""
FineTuningならこっちをつかう
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)#roiはregion of interestで関心領域でフィルタ処理や認識処理を適応したい画像中の部分的な関心領域のこと.畳み込みで得られたテンソルのことをfeaturemapと呼び、名前の通り、カーネルより抽出された特徴的な量のこと。カーネルは畳み込みで画像に何の数字をを掛け算するかを表した小さいサイズのマップ
num_classes: int = 2

in_features: int = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#backboneは入力画像の特徴を抽出する役割

"""

"""
モデルの修正および、違うbackboneを用いたい場合はこっちのコードを使う

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels: int = 1280
#AnchorはAnchorBoxの中心の座標、Anchorを囲うようにAnchor_Boxが作られる。
#AnchorBoxは検出したい物体を囲うように形成され、物体の候補領域を見つけに行く仕組み。scale(size)とaspect_ratio(四角の長さを変える比率)によってサイズが決まる
#Anchorgeneratorはbase_anchorを作成し、grid_size(画像中に設定するAnchorsの点数を示す。grid_size=(8,8)なら64個のAnchorが存在する)を定義、stride(Anchor同士の間隔)の定義をしそしてgrid_anchorsを作成
anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
#roi_poolingを行うことで入力の四角のサイズが異なっても同じサイズの特徴サイズに変換することができる。
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
"""

def get_model_instance_segmentation(num_classes):
#あらかじめ訓練されているインスタンスセグメンテーションのモデルをロード
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
#分類器の入力特徴量を設定
    in_features = model.roi_heads.box_predictor.cls_score.in_features
#あらかじめ学習されたヘッドを新しいものに置き換える
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#mask予測器の入力特徴量を取得
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
#mask予測器を変える
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      hidden_layer,
                                                      num_classes)
                                                      
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

"""
試しにモデルを使ってみる。

model = torchvision.models.detection.faster_rcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4
    collate_fn=utils.collate_fn)
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets =[{k: v for k, v in t.item()} for t in targets]
output = model(images, targets)

model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
"""


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes: int = 2

    dataset = PennFudanDataset('PennFudanPed/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed/PennFudanPed', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs: int = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch,
print_freq=10)
        lr_scheduler.step()
        
        evaluate(model, data_loader_test, device=device)
    print("That's it!")

if __name__ == "__main__":
    main()