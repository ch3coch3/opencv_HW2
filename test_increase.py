import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader
import os
from PIL import Image
from resnet import resnet50
from tqdm import tqdm 
from matplotlib import pyplot as plt
import random
import numpy as np

class customDataset(Dataset):
    def __init__(self,root_dir ,transform = None):
        self.transform = transform
        self.root_dir = root_dir
        self.dog_path = os.listdir("./kagglecatsanddogs_3367a/PetImages/Dog")
        self.cat_path = os.listdir("./kagglecatsanddogs_3367a/PetImages/Cat")
        # print(self.dog_path)
    def __len__(self):
        return len(self.dog_path) + len(self.cat_path) -2

    def __getitem__(self,index): 
        t = 0
        while t == 0:
            k = random.randint(0,1)
            if k == 1:
                file_size = os.path.getsize(os.path.join(self.root_dir,'Cat',self.cat_path[int(index/2)]))
                if file_size >= 10:
                    if self.cat_path[int(index/2)].endswith('.jpg'):
                        image = Image.open(os.path.join(self.root_dir,'Cat',self.cat_path[int(index/2)]))
                        if image.mode != "RGB" or self.cat_path[int(index/2)] == "11702.jpg":
                            index = index + 1
                            continue
                        t = 1
                        return (self.transform(image), torch.tensor(1))
                index = index + 1
            elif k == 0:
                file_size = os.path.getsize(os.path.join(self.root_dir,'Dog',self.dog_path[int(index/2)]))
                if file_size >= 10:
                    if self.dog_path[int(index/2)].endswith('.jpg'):
                        image = Image.open(os.path.join(self.root_dir,'Dog',self.dog_path[int(index/2)]))
                        if image.mode != "RGB" or self.dog_path[int(index/2)] == "11702.jpg":
                            index = index + 1
                            continue
                        t = 1
                        return (self.transform(image), torch.tensor(0))
                index = index + 1

 
def predict(img_path):
    net = torch.load('model.pth')
    # net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    # img = transform(img).unsqueeze(0)
    # img_ = img.to(device)
    outputs = net(img)
    _,predicted = torch.max(outputs,1)
    plt.show(img)
    plt.title(predicted[0])

def imshow(img):
    img = img / 2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # data preprocessor
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = customDataset("./kagglecatsanddogs_3367a/PetImages/",transform = transform)


    train_set, test_set = torch.utils.data.random_split(dataset,[20000,5000])
    train_loader = DataLoader(dataset=train_set, batch_size=32,shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True,num_workers=4)
    
    model = resnet50(num_classes = 2)
    model.load_state_dict(torch.load('model_increase.pth')['state_dict'])
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            outputs = model(image)
            _, predicted = torch.max(torch.abs(outputs), 1)
            choose = predicted[0].item()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(torchvision.utils.make_grid(image[0]*0.224+0.456).permute(1,2,0))
            if choose == 1:
                plt.title('class:'+'cat')
            elif choose == 0:
                plt.title('class:'+'dog')
            plt.show()
            break



    