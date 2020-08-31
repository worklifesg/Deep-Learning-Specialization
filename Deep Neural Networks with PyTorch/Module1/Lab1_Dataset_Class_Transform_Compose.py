# Objective: Construct a basic dataset by using PyTorch and learn how to apply basic transformations to it.

#Import Libraries
import torch
from torch.utils.data import Dataset
import torchvision

#The torch.manual_seed() is for forcing the random function to give the same number every time we try to recompile it.
torch.manual_seed(1)

## Class for dataset - Constructor with default values(_init_), Getter(_getitem_), Get Length(_len_)

class toy_set(Dataset):
    
    def __init__(self,length=100,transform=None):
        self.x=2*torch.ones(length,2)
        self.y=torch.ones(length,1)

        self.len=length
        self.transform=transform
    
    def __getitem__(self,index):
        sample=self.x[index],self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample
    
    def __len__(self):
        return self.len
    
# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.

our_dataset=toy_set()

with open('Lab1_Dataset_Class_Results.txt','a') as f:
    print('Class Toyset Test: \n',file=f)
    print("Our toy_set object: ", our_dataset,file=f)
    print("Value on index 0 of our toy_set object: ", our_dataset[0],file=f)
    print("Our toy_set length: ", len(our_dataset),file=f)
    print('\n',file=f)    

# Loop to print first 3 elements in dataset
with open('Lab1_Dataset_Class_Results.txt','a') as f:
    print('First 3 elements in dataset: \n',file=f)
for i in range(3):
    x,y=our_dataset[i]
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ',i,'; x: ',x,'; y: ',y,file=f)

# Transforms - Normalize or standardize Data

# We want to add 1 to x and multiply y by 2, so define another class - use only _init_ and _call_

class add_mult(object):

    def __init__(self,addx=1,muly=2):
        self.addx=addx # constructing x
        self.muly=muly # constructing y
    
    def __call__(self,sample):
        x=sample[0] #assigning x as 0th index
        y=sample[1] #assigning y as 1st index
        x=x+self.addx # adding function
        y=y*self.muly # multiplication function 
        sample=x,y # storing new values in sample
        return sample

## Using transform in Dataset as originally it was None.

a_m=add_mult() #calling transform function
data_set=toy_set() #calling dataset function

# Use loop to print out first 10 elements in dataset
with open('Lab1_Dataset_Class_Results.txt','a') as f:
    print('\n',file=f) 
    print('First 10 elements in dataset: \n',file=f)
for i in range(10):
    x, y = data_set[i]
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y,file=f)
    x_, y_ = a_m(data_set[i])
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_,file=f)

# Also, we can call transform function directly to dataset class
cust_data_set=toy_set(transform=a_m)

with open('Lab1_Dataset_Class_Results.txt','a') as f:
    print('-----------------------------------------------------------------',file=f)
    print('First 10 elements in dataset by directly transforming in dataset: \n',file=f)
for i in range(10):
    x, y = data_set[i]
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y,file=f)
    x_, y_ = cust_data_set[i]
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_,file=f)

#   Compose - use to have multiple transformations on the dataset object

from torchvision import transforms

#new class to multiply each element by 100

class mult(object):

    def __init__(self,mult=100):
        self.mult=mult
    
    def __call__(self,sample):
        x=sample[0] #assigning x as 0th index
        y=sample[1] #assigning y as 1st index
        x=x*self.mult 
        y=y*self.mult 
        sample=x,y # storing new values in sample
        return sample

data_transform=transforms.Compose([add_mult(),mult()]) #first will add x by 1 and multiply y by 2 and use to new result to multiply by 100

x,y=data_set[0]
x_,y_=data_transform(data_set[0])

#also can set it directly on Dataset
compose_data_set=toy_set(transform=data_transform)

with open('Lab1_Dataset_Class_Results.txt','a') as f:
    print('\n',file=f)    
    print('Data Transform (Multiple): \n',file=f)
    print( 'Original x: ', x, 'Original y: ', y,file=f)
    print( 'Transformed x_:', x_, 'Transformed y_:', y_,file=f)
    print('-----------------------------------------------------',file=f)    
    print('\n',file=f) 

# Use loop to print out first 3 elements in dataset (original dataset,custom data set, compose dataset)

for i in range(3):
    x, y = data_set[i]
    x_, y_ = cust_data_set[i]
    x_co, y_co = compose_data_set[i]
    with open('Lab1_Dataset_Class_Results.txt','a') as f:
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y,file=f)
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_,file=f)
        print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co,file=f)
        print('-----------------------------------------------------',file=f) 


