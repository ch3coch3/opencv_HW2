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
import numpy as np

# class customDataset(Dataset):
#     def __init__(self, train = True, transform=None):
#         cat = {}
#         dog = {}
#         self.cat = cat
#         self.dog = dog
#         self.transform = transform

#         # load the path
#         if train:

#             paths_dog =  os.listdir('./kagglecatsanddogs_3367a/PetImages/Dog')
#             paths_dog = [os.path.join('./kagglecatsanddogs_3367a/PetImages/Dog',i) for i in paths_dog]
#             dog = {}
#             for i in range(int(len(paths_dog)*0.9)):
#                 dog[paths_dog[i]] = 'dog'
#             paths_cat =  os.listdir('./kagglecatsanddogs_3367a/PetImages/Cat')
#             paths_cat = [os.path.join('./kagglecatsanddogs_3367a/PetImages/Cat',i) for i in paths_cat]
#             cat = {}
#             for i in range(int(len(paths_cat)*0.9)):
#                 cat[paths_cat[i]] = 'cat'

#             # self.root_dir = root_dir
#             # self.img_path = os.listdir(self.root_dir)
#         else:
#             paths_dog =  os.listdir('./kagglecatsanddogs_3367a/PetImages/Dog')
#             paths_dog = [os.path.join('./kagglecatsanddogs_3367a','PetImages','Dog',i) for i in paths_dog]
#             dog = {}
#             for i in range(int(0.9*len(paths_dog)),len(paths_dog)):
#                 dog[paths_dog[i]] = 'dog'
#             paths_cat =  os.listdir('./kagglecatsanddogs_3367a/PetImages/Cat')
#             paths_cat = [os.path.join('./kagglecatsanddogs_3367a','PetImages','Cat',i) for i in paths_cat]
#             cat = {}
#             for i in range(int(0.9*len(paths_cat)),len(paths_cat)):
#                 cat[paths_cat[i]] = 'cat'
#                 # print(cat)



#     def __getitem__(self,index):
        
#         # if st == 'cat':
#         #     image_cat = Image.open(cat[index])
#         #     label = 0
#         #     if self.transform:
#         #         image_cat = self.transform(image_cat)
#         #     return image_cat, label
#         # else:
#         #     image_dog = Image.open(dog[index])
#         #     label = 1
#         #     if self.transform:
#         #         image_dog = self.transform(image_dog)
#         #     return image_dog, label
#         image_dog = Image.open(self.dog[index])
#         label = 1
#         if self.transform:
#             image_dog = self.transform(image_dog)
#         return image_dog, label

#     def __len__(self):
#         return len(self.cat), len(self.dog)
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
        for root,_,files in os.walk(self.root_dir):
            for File in files:
                if File.endswith('.jpg'):
                    filename = os.path.join(root,File)
                    print(filename)
                    category_name = os.path.basename(root)
                    image = Image.open(filename)
                    if self.transform:
                        image = self.transform(image)
                    if category_name == 'Cat':
                        return (image, torch.tensor(1))
                    elif category_name == 'Dog':
                        return (image, torch.tensor(0))
 
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

if __name__ == '__main__':
    # # data preprocessor
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])

    dataset = customDataset("./kagglecatsanddogs_3367a/PetImages/",transform = transform)
    # # print(temp)
    # # dog_img = temp[0]
    # # dog_label = temp[1]
    # # cat_img = temp[2]
    # # cat_label = temp[3]

    train_set, test_set = torch.utils.data.random_split(dataset,[20000,5000])
    train_loader = DataLoader(dataset=train_set, batch_size=8,shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=True)
    dataiter = iter(test_loader)
    image,labels = dataiter.next()
    img = Image.open(image)
    
    # predict(image)
    # train_loader = DataLoader(dataset=train_set, batch_size=8,shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=True)

    