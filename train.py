import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import SubsetRandomSampler, Dataset
import os
# from torch.utils.tensorboard import SummaryWriter
from resnet import resnet50
from tqdm import tqdm 
from PIL import Image

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
                    category_name = os.path.basename(root)
                    image = Image.open(filename)
                    if self.transform:
                        image = self.transform(image)
                    if category_name == 'Cat':
                        return (image, torch.tensor(1))
                    elif category_name == 'Dog':
                        return (image, torch.tensor(0))

def train_step(model, loader):
    model.train()
    loss_seq = []
    outputs_list = []
    targets_list = []
    iterator = tqdm(loader, desc='Training:', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_seq.append(loss.item())
        loss.backward()
        optimizer.step()

        outputs_list.append(outputs)
        targets_list.append(targets)
    
    epoch_loss = sum(loss_seq) / len(loss_seq)
    train_loss_list.append(epoch_loss)
    
    targets = torch.cat(targets_list)
    outputs = torch.cat(outputs_list)
    acc = accuraccy(outputs, targets)        
    training_acc.append(acc)


    torch.save({'state_dict': model.state_dict()}, './model.pth')
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
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    # hyper parameters
    num_epochs = 5
    batch_size = 16
    learning_rate = 0.001

    # data preprocessor
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])

    dataset = customDataset("./kagglecatsanddogs_3367a/PetImages/",transform = transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset,[20000,5000])

    # train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True, transform=transform,download=True)
    # test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False, transform=transforms.ToTensor())
    # train_dataset = customDataset(train=True, transform=True)
    # test_dataset = customDataset(train=False, transform=True)

    # data loader
    # split = int(len(train_dataset)*0.9)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)



    model = resnet50()
    model.cuda()

    # # tensor board
    # writer = SummaryWriter()
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
    for epoch in range(num_epochs):
        print('---------- Epoch {} ------------'.format(epoch+1))            
        train_acc, train_loss = train_step(model, train_loader)
        # val_acc, val_loss = validate(model, val_loader, 'val')
        test_acc = validate(model, test_loader, 'test')
        print('Training loss: {:.3f}, training acc: {:.3f}; Testing acc: {:.3f}'.format(train_loss, train_acc, test_acc))            

