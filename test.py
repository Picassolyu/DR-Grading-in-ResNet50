import torch
from torchvision import models, transforms, datasets
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import os
from tqdm import tqdm

def main():
    # 定义一下变换
    image_transforms = {
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

    batch_size = 32
    classes = ["0", "1", "2"]  # 识别种类名称（顺序要与训练时的数据导入编号顺序对应，可以使用datasets.ImageFolder().class_to_idx来查看）

    dataset = 'data'
    test_directory = os.path.join(dataset, 'test')

    data = {
            'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test']),
        }

    test_data_size = len(data['test'])
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=8)




    transf = transforms.ToTensor()
    device = torch.device('cpu')
    num_classes = 3
    model_path = "ModelHistory/data_model_N6.pt"
    #image_input = cv2.imread("C. Diabetic Retinopathy Grading/1. Original Images/b. Testing Set/473.png")
    #image_input = transf(image_input)
    #image_input = torch.unsqueeze(image_input, dim=0)
    # 搭建模型
    resnet50 = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    for param in resnet50.parameters():
        param.requires_grad = False

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    resnet50 = torch.load(model_path)

    f2 = open('probabilities.txt', 'a+')

    for i, (inputs, labels) in enumerate(tqdm(test_data)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = resnet50(inputs)
        print("标签值：", labels)
        print(outputs, file=f2)


    #outputs = resnet50(image_input)
    #value, id = torch.max(outputs, 1)
    #print(outputs, "\n", "结果是：", classes[id])

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!