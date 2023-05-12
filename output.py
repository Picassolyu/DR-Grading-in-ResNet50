import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.nn as nn
import os
from PIL import Image
# import main
import csv

# model = main.get_model()
model_path = 'ModelHistory/data_model_N7.pt'
test_pth = 'C. Diabetic Retinopathy Grading/1. Original Images/b. Testing Set/'
model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)
model = torch.load(model_path)


# model = models.resnet50(pretrained=False)
# pthfile = './model_data.pth'
# saved_model = torch.load(pthfile)
# model_dict = model.state_dict()
# state_dict = {k:v for k,v in saved_model.items() if k in model_dict.keys()}
# print(state_dict.keys())
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)
# print(model)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(256),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Test_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.root_dir)[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image,img_name


dataset = Test_data(root_dir=test_pth, transform=transform)
batch_size = 50
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

with open('predictions2.csv', 'w', newline='\n') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['case', 'class', 'P0', 'P1', 'P2'])
    for i, (data,imgname) in enumerate(dataloader):
        images = data.to(device)
        outputs = model(images)
        prob = torch.softmax(outputs, dim=1)
        _, preds = prob.max(dim=1)
        pred = preds.item()
        p0 = prob[0, 0].item()
        p1 = prob[0, 1].item()
        p2 = prob[0, 2].item()
        image_names = ''.join(imgname)
        image_names = image_names.replace('(', '').replace(')', '').replace('\'', '')
        writer.writerow([image_names, pred, p0, p1, p2])
        print(image_names, pred, p0, p1, p2)

