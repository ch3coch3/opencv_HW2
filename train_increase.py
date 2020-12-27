import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import SubsetRandomSampler, Dataset
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from resnet import resnet50
from tqdm import tqdm 
from PIL import Image

class customDataset(Dataset):
    def __init__(self,root_dir ,transform = None):

        self.transform = transform
        self.root_dir = root_dir
        self.dog_path = os.listdir("./kagglecatsanddogs_3367a/PetImages/Dog")
        self.cat_path = os.listdir("./kagglecatsanddogs_3367a/PetImages/Cat")
        print((len(self.dog_path) + len(self.cat_path) -2)*4)
    def __len__(self):
        return (len(self.dog_path) + len(self.cat_path) -2)*4

    def __getitem__(self,index): 
        
        # for i in range(4):
        a = random.randint(20,30)
        b = random.randint(20,30)
        transform = transforms.Compose([
            # transforms.Grayscale(1),
            # transforms.ToPILImage(mode='RGB'),
            transforms.Resize([224+a,224+b]),
            # transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform = transform
        t = 0
        index = int(index/8)
        while t == 0:
            k = random.randint(0,1)
            if index <= (len(self.dog_path) + len(self.cat_path) -2)*4:
                if k == 1:
                    file_size = os.path.getsize(os.path.join(self.root_dir,'Cat',self.cat_path[index]))
                    if file_size >= 10:
                        if self.cat_path[index].endswith('.jpg'):
                            image = Image.open(os.path.join(self.root_dir,'Cat',self.cat_path[index]))
                            if image.mode != "RGB" or self.cat_path[index] == "11702.jpg":
                                index = index + 1
                                continue
                            t = 1
                            return (self.transform(image), torch.tensor(1))
                    index = index + 1
                elif k == 0:
                    # print(index)
                    file_size = os.path.getsize(os.path.join(self.root_dir,'Dog',self.dog_path[index]))
                    if file_size >= 10:
                        if self.dog_path[index].endswith('.jpg'):
                            image = Image.open(os.path.join(self.root_dir,'Dog',self.dog_path[index]))
                            if image.mode != "RGB" or self.dog_path[index] == "11702.jpg":
                                index = index + 1
                                continue
                            t = 1
                            return (self.transform(image), torch.tensor(0))
                    index = index + 1
            else:
                index = index - 2

def train_step(model, loader):
    global time
    model.train()
    loss_seq = []
    outputs_list = []
    targets_list = []
    # j = 1
    iterator = tqdm(loader, desc='Training:', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_seq.append(loss.item())
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_accuracy",accuraccy(outputs,targets),time)
        writer.add_scalar("train_loss", loss,time)
        outputs_list.append(outputs)
        targets_list.append(targets)
        time = time + 1
    
    epoch_loss = sum(loss_seq) / len(loss_seq)
    train_loss_list.append(epoch_loss)
    
    targets = torch.cat(targets_list)
    outputs = torch.cat(outputs_list)
    acc = accuraccy(outputs, targets)        
    training_acc.append(acc)


    torch.save({'state_dict': model.state_dict()}, './model_increase.pth')
    return acc, epoch_loss

def validate(model, loader, state='val'):
    model.eval()
    loss_seq = []
    outputs_list = []
    targets_list = []
    
    with torch.no_grad():
        desc_word = 'Validating' if state == 'val' else 'Testing'
        iterator = tqdm(loader, desc=desc_word, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            targets_list.append(targets)
            outputs_list.append(outputs)
            loss_seq.append(loss.item())
        
        targets = torch.cat(targets_list)
        outputs = torch.cat(outputs_list)
        acc = accuraccy(outputs, targets)
        
        if state == 'val':
            epoch_loss = sum(loss_seq) / len(loss_seq)
            val_loss_list.append(epoch_loss)
            val_acc.append(acc)

            return acc, epoch_loss
        else:
            testing_acc.append(acc)
    
            return acc

def accuraccy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = float(predictions.eq(targets).cpu().sum())
    acc = 100 * correct / targets.size(0)

    return acc

if __name__ == '__main__':
    
    global time
    time = 1
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    # hyper parameters
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    
    
    # data preprocessor
    transform = transforms.Compose([
        # transforms.Grayscale(1),
        # transforms.ToPILImage(mode='RGB'),
        transforms.Resize([224,224]),
        # transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = customDataset("./kagglecatsanddogs_3367a/PetImages/",transform = transform)
    print(type(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,[20000*4,5000*4])

    # train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True, transform=transform,download=True)
    # test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False, transform=transforms.ToTensor())
    # train_dataset = customDataset(train=True, transform=True)
    # test_dataset = customDataset(train=False, transform=True)

    # data loader
    # split = int(len(train_dataset)*0.9)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,num_workers=4)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,num_workers=4)



    model = resnet50(num_classes = 2)
    model.cuda()

    # # tensor board
    writer = SummaryWriter("runs/increase")
    # image, labels = next(iter(train_loader))
    # grid = torchvision.utils.make_grid(image)
    # writer.add_image('images',grid,0)
    # writer.add_graph(model,image)
    # writer.close()
    
    # Loss and optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    training_acc = []
    val_loss_list = []
    val_acc = []
    testing_acc = []
    f = open('training_increase_result.txt','w')
    for epoch in range(num_epochs):
        print('---------- Epoch {} ------------'.format(epoch+1))            
        train_acc, train_loss = train_step(model, train_loader)
        # writer.add_scalar("train_accuracy",train_acc,epoch)
        # writer.add_scalar("train_loss", train_loss,epoch)
        # val_acc, val_loss = validate(model, val_loader, 'val')
        test_acc = validate(model, test_loader, 'test')
        f.write('Training loss: {:.3f}, training acc: {:.3f}; Testing acc: {:.3f} \n'.format(train_loss, train_acc, test_acc))
        print('Training loss: {:.3f}, training acc: {:.3f}; Testing acc: {:.3f}'.format(train_loss, train_acc, test_acc))            
    f.close()
    writer.close()
