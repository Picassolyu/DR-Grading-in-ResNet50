# 该文件用于将图片根据对应标签放入指定的文件夹

import pandas as pd
import os
import shutil

# 读取文件
file = open("C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv", "rb")
list = pd.read_csv(file)
list["FILE_PNG"] = ""
list["FILE_ID"] = list["image name"] + list["FILE_PNG"]

# 创建文件夹，进行分类
for i in range(3):
    label_dir = os.path.join("{:2}".format(i))
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    listnew = list[list["DR grade"] == i]
    list_image = listnew["FILE_ID"].tolist()
    for img in list_image:
        shutil.copy("C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set/" + img, label_dir)