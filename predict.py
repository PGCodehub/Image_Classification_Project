#Created by pramod G

#All imports
from time import time,sleep
import argparse

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import json

#for keeping workspace live during non activity
from workspace_utils import active_session

# Main program function defined below
def main():
    
    start_time = time()
    
    #taking the input
    in_arg = get_input_args()
    
    #print(in_arg.hidden_units, in_arg.dir)
    
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
    
    #Load the desired model 
    loaded_model = loadCheckpoint(in_arg.checkpointpath,device)
    
    #print("Accuracy of newly loaded model :")
    #Acurracy(loaded_model,device)

    #read catagories names
    catnames = in_arg.category_names
    with open(catnames, 'r') as f:
        cat_to_name = json.load(f)
        
    #preprocess the data
    probs, classes = predict(in_arg.img, loaded_model, device , in_arg.top_k ) 
    
    probs = probs.cpu()
    probs = probs.numpy()
    Topprob = probs.argmax()
    
    print("The most likely image class is {} and it's associated probability is {}".format(classes[Topprob],probs[0][Topprob]))
    
    print("The top {} classes along with associated probabilities are {}, {}".format(in_arg.top_k ,classes,probs))
    
    TopKClasses = [cat_to_name[clas] for clas in classes]
          
    for i,clas in enumerate(TopKClasses):
          print(" The top {} class probably is {}".format(i+1,clas))
    

    end_time = time() 
    ttime = end_time - start_time
    hh = round(ttime/(3600))
    mm = round((ttime%3600)/60)
    ss = round((ttime%3600)%60)
    tot_time = "{}:{}:{}".format(hh,mm,ss)
    print("\n** Total Elapsed Runtime:", tot_time)


def get_input_args():
    parser = argparse.ArgumentParser(description = " This program is for predicting when given a image by specifying path to Imagefile and a checkpoint to create model from ")
    parser.add_argument( 'img',type = str, default = 'flowers/test/100/image_07899.jpg',  metavar='' , help = 'This is to specify Path to the imagefile need to be predicted (default if running from workspace- \'paind-project/flowers/test/100/image_07899.jpg\')')
    parser.add_argument( 'checkpointpath', type = str, default = 'CLcheckpoint.pth', metavar='' , help = 'This is to specify  checkpoint from which tained model will be loaded (default-\'CLcheckpoint.pth\')')
    parser.add_argument( '----category_names',type = str, default = 'cat_to_name.json', metavar='' , help = 'This is to specify to load a JSON file that maps the class values to other category names (default-\'cat_to_name.json.pth\')')
    parser.add_argument( '--top_k', type = int, default = 5, metavar='' , help = 'This is to specify no of top probabilities you want to predict (default-\'5\')')
    
    #for flags like to be trained on gpu or not like that
    group = parser.add_mutually_exclusive_group()
    group.add_argument( '--gpu',  action = 'store_true' , help = 'This is to set the flag if you want to use gpu or not(default if not specified-\'cpu\'')
    
    arg = parser.parse_args()
    
    return arg

def preprocess(dir):
    #transforms for  testing sets
    Pre_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image = Image.open(dir)
    

    # Load the datasets with ImageFolder
    PreImage = Pre_transforms(image)[:3,:,:]
    
    return PreImage 

def predict(image_path, model , device ,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = preprocess(image_path)
    
    #print(image.shape)

    image.unsqueeze_(0)
    
    #print(image.shape)
    
    image = image

    image = image.to(device)
    
    #model.type(torch.DoubleTensor)
    
    model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        
        output = model.forward(image)
        
        prob = torch.exp(output)
        
        probs , indices = torch.topk(prob , topk) 
        
        indice = indices.cpu()

        indice = indice.numpy()[0]
        
        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}  
        
        #print(idx_to_class)
        
        classes = [idx_to_class[index] for index in indice]
        
    
    return  probs , classes
                        
                        
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
