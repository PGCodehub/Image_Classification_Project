#Created by pramod G

#All imports
from time import time,sleep
import argparse

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from os import path , makedirs 

#for keeping workspace live during non activity
from workspace_utils import active_session

# Main program function defined below
def main():
    
    start_time = time()
    
    #taking the input
    in_arg = get_input_args()
    
    #print(in_arg.hidden_units, in_arg.dir)

    
    #preprocess the data
    train_dataloaders, valid_dataloaders , test_dataloaders  , train_datasets = preprocess(in_arg.dir)
    
    #Load the desired model
    model = PretrainedModel(in_arg.arch)
    
    #set the final fully connected layer 
    model = ClassifierFC(model,in_arg.arch, in_arg.hidden_units)
    
    #Set which device to run on 
    if in_arg.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("Sorry but there is no gpu available" )
            print("Program is terminated")
            exit()       
    else:
        device = torch.device("cpu")
        
            
                
            
    
    #Set all the hyperparameters
    criterion = nn.NLLLoss()
    if(in_arg.arch == 'resnet50'):
        optimizer = optim.Adam(model.fc.parameters(), lr = in_arg.learning_rate)   
    if(in_arg.arch == 'densenet121'):
        optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)   
    if(in_arg.arch == 'vgg16'):
        optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)
        
    
    #Finally train the model
    
    print("Model is going to be trained on {} Dataset on {} Architecture with modified FC layer with hidden layer {} with learning rate of {} for {} Epochs on a {}..".format(in_arg.dir,in_arg.arch,in_arg.hidden_units,in_arg.learning_rate,in_arg.epochs,device))
    
    with active_session():
        model = trainthemodel(model,optimizer,criterion,in_arg.epochs,train_dataloaders,valid_dataloaders,device)
        #Acuuracy of Trainedmodel
        old_acuraccy = Acurracy(model,criterion,test_dataloaders,device)
    
        #savingcheckpoint and load
        if in_arg.save_dir != None:
            if not path.exists(in_arg.save_dir):
                makedirs(in_arg.save_dir)
            filename = in_arg.save_dir + '/CLcheckpoint.pth'
        else:
            filename = 'CLcheckpoint.pth'
        
        savecheckpoint(model,filename ,in_arg.arch ,in_arg.epochs,in_arg.learning_rate , train_datasets)
        
        loaded_model = loadCheckpoint(filename,device)
    
        

        #checking is model is saved and can load properly

        print("Accuracy of newly loaded model :")

        new_accuracy = Acurracy(model,criterion,test_dataloaders,device)

        if old_acuraccy == new_accuracy:
            print("Accuracy seems to match! Modelsaved and  loaded successfully!")
        else:
            print("Loading Model Unsuccessfull :(")
        
    print("Model Trained ,  Modelsaved in {}".format(filename))
 
    end_time = time() 
    ttime = end_time - start_time
    hh = round(ttime/(3600))
    mm = round((ttime%3600)/60)
    ss = round((ttime%3600)%60)
    tot_time = "{}:{}:{}".format(hh,mm,ss)
    print("\n** Total Elapsed Runtime:", tot_time)


def get_input_args():
    parser = argparse.ArgumentParser(description = " This program is for training on given dataset by specifying path to file and architecture to train on ")
    parser.add_argument( 'dir',type = str, default = 'paind-project/flowers/',  metavar='' , help = 'This is to specify Path to the dataset of image files need to be trained on(default if running from workspace- \'paind-project/flowers/\')')
    parser.add_argument( '--arch', type = str, default = 'resnet50', metavar='' , help = 'This is to specify  CNN model architecture to use for image classification(default- resnet50 )--->pick any of the following vgg16, resnet50 , densenet121 ')
    parser.add_argument( '--save_dir', type = str, default = None, metavar='' , help = 'This is to specify  directory in which tained model will be saved as a CLcheckpoint.pth(default- None )')
    parser.add_argument( '--learning_rate', type = float, default = 0.01, metavar='' , help = 'This is to specify which directory tained model will be saved(default-\'0.01\')')
    parser.add_argument( '--hidden_units',type = int, nargs = '+', default = [512], metavar='' , help = 'This is to specify no of hidenunits in each hiden layer, specify in this format for multiple hidden layers --->2048 512 102 <--   (default-\'[512]\')')
    parser.add_argument( '--epochs', type = int, default = 2, metavar='' , help = 'This is to specify no of epochs to run for the training(default-\'2\')')
    
    #for flags like to be trained on gpu or not like that
    group = parser.add_mutually_exclusive_group()
    group.add_argument( '--gpu',  action = 'store_true' , help = 'This is to set the flag if you want to use gpu or not(default if not specified-\'cpu\'')
    
    arg = parser.parse_args()
    
    return arg

def preprocess(dir):
    train_dir = dir + '/train'
    valid_dir = dir + '/valid' 
    test_dir = dir + '/test'
    
     #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 


    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,test_transforms)
    test_datasets = datasets.ImageFolder(test_dir,test_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle= True )
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle= True )
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32 )
    
    return train_dataloaders, valid_dataloaders , test_dataloaders , train_datasets

def PretrainedModel(name):
    model = models.__dict__[name](pretrained=True)
        
    return model

def ClassifierFC(model ,arch , hidden_layers):
    
    for param in model.parameters():
        param.requires_grad = False
    
    if(arch == 'resnet50'):
        fcclassifier = Network(2048,hidden_layers,102)
        model.fc = fcclassifier
        
    if(arch == 'densenet121'):
        fcclassifier = Network(1024,hidden_layers,102)
        model.classifier = fcclassifier
        
    if(arch == 'vgg16'):
        fcclassifier = Network(25088,hidden_layers,102)
        model.classifier = fcclassifier
        
    return model
        
#Validation 
def Validation(model,criterion,valid_dataloaders,device):
    testloss = 0
    accuracy = 0
    totaldatasetlen = 0
    model.to(device)

    for images , labels in valid_dataloaders:
        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)
        testloss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        
        equity = (labels == ps.max(dim = 1)[1])
        totaldatasetlen +=labels.size(0)
        accuracy += equity.type(torch.FloatTensor).mean()
    
    
    return testloss,accuracy,totaldatasetlen

#train the network
def trainthemodel(model,optimizer,criterion,epochs,train_dataloaders,valid_dataloaders,device):
    printevery = 40
    running_loss = 0 
    steps = 0
    
    model.to(device)

    print("The training about to start and print every {} steps...".format(printevery))
    print("The device its running on is : {}. ..".format(device))


    for e in range(epochs):
        model.train()
        
        for images,labels in train_dataloaders:
            steps += 1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            Loss = criterion(output,labels)
            
            Loss.backward()
            optimizer.step()
            
            running_loss += Loss.item()
            
            
            if steps%printevery == 0:
                model.eval()
                with torch.no_grad():
                    test_loss,accuracy,totaldatalen = Validation(model,criterion,valid_dataloaders,device)
                
                print("The no of steps is {}...".format(steps),
                      "The epoch is {}/{}....".format(e+1 ,epochs),
                      "The Loss of model is {:.4f}...".format(running_loss/printevery),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
                      "Test Accuracy: {:.3f}%".format(accuracy/len(valid_dataloaders) * 100)
                 )
                
                running_loss = 0
                model.train()  
                
                
    return model        
    print("Training Complete")

    
# TODO: Do validation on testsekt

def Acuuracyontestset(model,criterion,test_dataloaders,device):
    testloss = 0
    accuracy = 0
    totaldatasetlen = 0
    
    model.to(device)
    
    model.eval()
    for images , labels in test_dataloaders:
        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)        
        ps = torch.exp(output)
        
        equity = (labels == ps.max(dim = 1)[1])
        totaldatasetlen +=labels.size(0)
        accuracy += equity.type(torch.FloatTensor).sum().item()
    
    
    return accuracy,totaldatasetlen

def Acurracy(model,criterion,test_dataloaders,device):
    print("Calculating Acuracy:")
    with torch.no_grad():
        accuracy,total = Acuuracyontestset(model,criterion,test_dataloaders,device)
    
    print('Accuracy of the network on the {} test images: {}%...'.format(total,(100 * accuracy / total)))
    
    return (100 * accuracy / total)

# TODO: Save the checkpoint 
def savecheckpoint(model,filename ,arch ,epochs,learningrate , train_datasets):
    if(arch == 'resnet50'):
        modelcheckpoint = { 'input_size':2048,
                  'output_size':102,
                   'epochs': epochs,
                    'learningrate': learningrate,
                   'model' : arch,
                   'classifier' : model.fc,
                   'state_Dict': model.state_dict(),
                   'class_to_idx': train_datasets.class_to_idx
                   } 
    if(arch == 'densenet121'):
        modelcheckpoint = { 'input_size':1024,
                  'output_size':102,
                   'epochs': epochs,
                    'learningrate': learningrate,
                   'model' : arch,
                   'classifier' : model.classifier,
                   'state_Dict': model.state_dict(),
                   'class_to_idx': train_datasets.class_to_idx
                   }   
    if(arch == 'vgg16'):
        modelcheckpoint = { 'input_size':25088,
                  'output_size':102,
                   'epochs': epochs,
                    'learningrate': learningrate,
                   'model' : arch,
                   'classifier' : model.classifier,
                   'state_Dict': model.state_dict(),
                   'class_to_idx': train_datasets.class_to_idx
                   }
    
    torch.save(modelcheckpoint, filename)
    print("The model is saved in {} file..".format(filename))

    
#function that loads a checkpoint and rebuilds the model
def loadCheckpoint(filepath,device):
    
    if device == 'cuda:0':
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    arch = checkpoint['model']
    model = models.__dict__[arch](pretrained=True)
    if(arch == 'resnet50'):
        model.fc = checkpoint['classifier']   
    if(arch == 'densenet121'):
        model.classifier = checkpoint['classifier']   
    if(arch == 'vgg16'):
        model.classifier = checkpoint['classifier']
    
    
    model.load_state_dict(checkpoint['state_Dict'])
    #optimizer = checkpoint['optimizer_state_dict']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    
    for param in model.parameters():
        param.requires_grad = False
        
        #model.to()
    return model
    print("The model is loaded from {} file..".format(filepath))

        
        
        
        
        
#Defining Classifier with network class style
class Network(nn.Module):
    def __init__(self,input_size, hiddenlayers, output_size, drop_p = 0.5):
        
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hiddenlayers[0])])
        
        layersizes = zip(hiddenlayers[:-1],hiddenlayers[1:])
                
        #print(layersizes)
        
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layersizes])
        
        self.output = nn.Linear(hiddenlayers[-1],output_size)
                
        self.dropout = nn.Dropout(p = drop_p)
        
    def forward(self , x):
        
        for lin in self.hidden_layers:
            x = F.relu(lin(x))
            x = self.dropout(x)
            
        x = self.output(x)
        
        return F.log_softmax(x, dim= 1)
   
if __name__ == "__main__":
    main()
