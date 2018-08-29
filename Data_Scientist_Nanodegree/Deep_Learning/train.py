import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import argparse
import copy
from copy import deepcopy

                   
def load_transform_data(filename):
    data_dir = filename
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return train_data, trainloader, validloader, testloader

def label_mapping():
    with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def pre_trained_model(architecture):
    #print(architecture)
    if architecture is not None:
        model = getattr(models, architecture)(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    return model

def model_parameters(model, learning_rate, hidden_units, cat_to_name):
# Freeze parameters so we don't backprop through them- vgg16
    for param in model.parameters():
        param.requires_grad = False

    #print(type(model.classifier))
    if type(model.classifier) is torch.nn.modules.linear.Linear:
        input_size= model.classifier.in_features 
    else:
        input_size=model.classifier[0].in_features 
    output_size=len(cat_to_name)
    if hidden_units is not None:
        NN_layer=OrderedDict()
        for i in range(0,len(hidden_units)):            
            if i==0:
                NN_layer.update({"fc{}".format(i):nn.Linear(input_size, hidden_units[i])})
                NN_layer.update({"dropout":nn.Dropout(p=0.2)})
            else:
                NN_layer.update({"fc{}".format(i): nn.Linear(hidden_units[i-1], hidden_units[i])})
            if i!= len(hidden_units):
                NN_layer.update({"relu{}".format(i):nn.ReLU()})                
            #print(NN_layer)
        NN_layer.update({"fc{}".format(i+1): nn.Linear(hidden_units[-1], output_size)})
        NN_layer.update({'output': nn.LogSoftmax(dim=1)})                
            
                
        #print(NN_layer)    
        classifier=nn.Sequential(NN_layer)    
        #print(classifier)
    else:
        classifier = nn.Sequential(OrderedDict([ 
                          ('fc1', nn.Linear(model.classifier[0].in_features, 1000)),
                          ('dropout1',nn.Dropout(p=0.2)),
                          ('relu1', nn.ReLU()),                          
                          ('fc2', nn.Linear(1000, 512)),
                          ('relu2', nn.ReLU()),                          
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    if learning_rate is not None:
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    else:    
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    return model.classifier, criterion, optimizer

def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    valid_loss = 0
    
    for images, labels in validloader:
         
        #images.resize_(images.shape[0], 784)
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy
    
def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    if device == 'gpu':
        # change to cuda
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
                
def validation_accuracy(model, validloader, criterion):
    model.eval()
# Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        valid_loss, accuracy = validation(model, validloader, criterion)
    print("Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
             
def testing_accuracy(model, testloader, criterion):
    model.eval()
# Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion)
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(testloader))) 
   
             
def save_model (train_data, model, filename, architecture):
    model.to('cpu')
    model.class_to_idx = train_data.class_to_idx
    if architecture is None:
        architecture='vgg16'
    checkpoint = {
    'tag': 'model',
    'model': copy.deepcopy(model),
    'classifier': model.classifier,
    'architecture': architecture,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
    }
    if filename is not None:
        torch.save(checkpoint,filename)
    else:
        torch.save(checkpoint,'part2_checkpoint_updated.pth')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help='Dataset directory')
    parser.add_argument("--save_dir", help='Path to save checkpoint')
    parser.add_argument("--arch", help='Choose architecture')
    parser.add_argument("--learning_rate", help='Choose learning rate', type=float)
    parser.add_argument("--hidden_units", help="Input the number of hidden units in each layer", type=int, nargs='+')
    parser.add_argument("--epochs", help='Choose number of epochs', type=int)
    parser.add_argument("--gpu", help='Choose GPU for training')
    args = parser.parse_args()
    #print(args.filename)
                          
    train_data, trainloader, validloader, testloader =  load_transform_data(args.filename)      
    cat_to_name = label_mapping()
    model = pre_trained_model(args.arch)
    model.classifier, criterion, optimizer = model_parameters(model, args.learning_rate, args.hidden_units, cat_to_name)
    if args.epochs is not None:
        do_deep_learning(model, trainloader, args.epochs, 40, criterion, optimizer, args.gpu)
    else:    
        do_deep_learning(model, trainloader, 3, 40, criterion, optimizer, args.gpu)
    validation_accuracy(model, validloader, criterion)
    testing_accuracy(model, testloader, criterion)
    save_model(train_data, model, args.save_dir, args.arch)                 