from PIL import Image
mask = Image.open(r'C:\Users\wakabayashi kenta\repo\python-practice\PennFudanPed\PennFudanPed\PNGImages\FudanPed00001.png')
# 各マスクのインスタンスは、0からNまでの異なる色を持っています。Nはインスタンスの数です。
# 簡単に可視化するために、マスクにカラーパレットを追加しましょう。
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
mask