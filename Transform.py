# -*- coding: UTF-8 -*-

import glob
import os
from PIL import Image

output_path = 'output'  # 输出文件夹名称

img_list = []
img_list.extend(glob.glob('*.png'))  # 所有png图片的路径

#print(img_list)  # 打印查看是否遍历所有图片

for img_path in img_list:
    img_name = os.path.splitext(img_path)[0]  # 获取不加后缀名的文件名
    print(img_name)  # 打印查看文件名
    im = Image.open(img_path)
    im = im.convert("RGB")  # 把PNG格式转换成的四通道转成RGB的三通道
    im_rotate = im.rotate(180, expand=True)  # 逆时针旋转90度,expand=1表示原图直接旋转
    # 判断输出文件夹是否已存在，不存在则创建。
    folder = os.path.exists(output_path)
    if not folder:
        os.makedirs(output_path)
    # 把旋转后的图片存入输出文件夹
    im_rotate.save(output_path + '/' + img_name+'_rotated_180'+'.png')

print('所有图片均已旋转完毕，并存入输出文件夹')
