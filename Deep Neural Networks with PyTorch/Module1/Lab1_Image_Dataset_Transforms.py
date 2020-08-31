# In this lab, you will build a dataset objects for images; many of the processes can be applied to a larger dataset. 
# Then you will apply pre-build transforms from Torchvision Transforms to that dataset (Auxillary functions,TorchVision Transforms)

#Import libraries
import torch
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import os

from torch.utils.data import Dataset,DataLoader
from matplotlib.pyplot import imshow
from PIL import Image

torch.manual_seed(0)

# General function to be used 
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

#### Auxillary functions

# Read CSV file from the URL and print out the first five samples
directory=''
csv_file='index.csv'
csv_path=os.path.join(directory,csv_file)

df = pd.read_csv(csv_path)
with open('Lab1_Image_Dataset_Results.txt','a') as f:
    print(df.head(),file=f) #reading the file
    print('File name for 1st row:', df.iloc[0,1],file=f) #obtaining the file name of row 0 and column 1 
    print('Class value for 1st row, y: ',df.iloc[0,0],file=f)
    print('File name for 2nd row:', df.iloc[1,1],file=f) #obtaining the file name of row 1 and column 1 
    print('Class value for 2nd row, y: ',df.iloc[1,0],file=f)
    print('The total number of rows: ',df.shape[0],file=f)
    print('\n',file=f)

#Load Image - need directory and image name

image_name_2=df.iloc[1,1] # let us say we want second image in the list
image_name_20=df.iloc[19,1] #20th picture

image_path_2=os.path.join(directory,image_name_2) # find iamge path same as we did to read url above
image_path_20=os.path.join(directory,image_name_20)

image_2=Image.open(image_path_2) # open the image using PIL image function
image_20=Image.open(image_path_20)

with open('Lab1_Image_Dataset_Results.txt','a') as f:
    print('Image_name 2: ',image_name_2,file=f) 
    print('Image_path 2: ',image_path_2,file=f) 
    print('Image_name 20: ',image_name_20,file=f) 
    print('Image_path 20: ',image_path_20,file=f) 


fig, axs = plt.subplots(2, 2)
fig.suptitle('Images')
axs[0, 0].imshow(image_2,cmap='gray',vmin=0,vmax=255)
axs[0, 0].set_title(df.iloc[1, 0])
axs[0, 1].imshow(image_20,cmap='gray',vmin=0,vmax=255)
axs[0, 1].set_title(df.iloc[19, 0])

### Creating Dataset class for Images

class Dataset(Dataset):

    def __init__(self,csv_file,data_dir,transform=None):
        self.data_dir=data_dir #image directory

        self.transform=transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        self.data_name=pd.read_csv(data_dircsv_file) #load csv file containing image info

        self.len=self.data_name.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):

        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx,1]) #image file path
        img=Image.open(img_name) #open image file

        y=self.data_name.iloc[idx,0] #class or label name

        if self.transform:
            img=self.transform(img)
        
        return img,y

##calling dataset class

dataset=Dataset(csv_file=csv_file,data_dir=directory) #creating dataset objects usinng csv file and directory

img_2=dataset[0][0]
y_2=dataset[0][1]
img_9=dataset[9][0]
y_9=dataset[9][1]


axs[1, 0].imshow(img_2,cmap='gray',vmin=0,vmax=255)
axs[1, 0].set_title(df.iloc[0, 0])
axs[1, 1].imshow(img_9,cmap='gray',vmin=0,vmax=255)
axs[1, 1].set_title(df.iloc[9, 0])

plt.show()
