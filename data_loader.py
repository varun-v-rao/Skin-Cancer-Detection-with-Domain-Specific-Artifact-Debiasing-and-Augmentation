import os
import random
from random import shuffle
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image 
from numpy.random import choice
import albumentations as albu
import cv2
from globalbaz import args
from sklearn.model_selection import train_test_split


def hair_mask(hairs,IMAGE_SIZE):
           
    mask_to_chose = choice(np.arange(14), 1,p=[0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.075,0.075,0.075,0.075])[0]
    mask = hairs[mask_to_chose]
    hair_trans = albu.Compose([albu.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], shift_limit=[-0.1,0.15],border_mode=3,p=1.)])
    mask = hair_trans(image = mask)['image']
    mask = cv2.resize(mask/255,(IMAGE_SIZE,IMAGE_SIZE),cv2.INTER_CUBIC)
    mask[mask == 1.] =  255
    mask[mask != 255.] = 0

    return mask


def mm_aug(mms,no_mm):
    mask_to_chose = choice(np.arange(10), 1,p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
    final_mask = mms[mask_to_chose]
    graymask = cv2.cvtColor(final_mask,cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.bitwise_not(cv2.threshold(graymask,30,255,cv2.THRESH_BINARY)[1])
    no_mm_sub = cv2.bitwise_and(no_mm,no_mm,mask=thresh1)
    new_img = cv2.add(no_mm_sub,final_mask)

    return new_img


def ruler_aug(ruler,IMAGE_SIZE):
    mask_to_chose = choice(np.arange(10), 1,p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
    mask = ruler[mask_to_chose]
    ruler_trans = albu.Compose([albu.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], shift_limit=[-0.1,0.15],border_mode=3,p=1.)])
    mask = ruler_trans(image=mask)['image']
    mask = cv2.resize(mask/255,(IMAGE_SIZE,IMAGE_SIZE),cv2.INTER_CUBIC)
    mask[mask == 1.] =  255
    mask[mask != 255.] = 0
    return mask
    
    

class ImageFolder(data.Dataset):
    def __init__(self, csv,image_size=256,mode='train',augmentation_prob=1.0,transform=None, transform2=None):
        """Initializes image paths and preprocessing module."""
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        self.transform2 = transform2
        self.args = args
        # GT : Ground Truth
        
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        #print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        row = self.csv.iloc[index]
        '''
        image_path = row.filepath
        
        filename = row.image_name
        
        GT_path = row.mm_masks
        
        image = Image.open(image_path)
        print(image.size)
        try:
            GT = Image.open(GT_path)
        except:
            GT = Image.open(image_path)

        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        ResizeRange = random.randint(300,320)
        Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
        p_transform = random.random()
        
        data = torch.tensor(np.array(cv2.imread((image_path)))).float()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
						
            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
			
            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0,20)
            ShiftRange_upper = random.randint(0,20)
            ShiftRange_right = image.size[0] - random.randint(0,20)
            ShiftRange_lower = image.size[1] - random.randint(0,20)
            image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

            image = Transform(image)

            Transform =[]


            Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
    		
            image = Transform(image)
            GT = Transform(GT)
            
            #aug_prob = random.randint(1,2)
            aug_prob = 1
    
            if aug_prob == 1: #hair augmentation
                hairs = np.load('hair_array.npy')
                msk = hair_mask(hairs,IMAGE_SIZE=256).astype(np.uint8)
                image = cv2.bitwise_and(image,image,mask= msk)    
    
    
            if aug_prob == 2: #ruler augmentation
                ruler = np.load('ruler_array.npy')
                msk = ruler_aug(ruler,IMAGE_SIZE=256).astype(np.uint8)
                final_msk = cv2.bitwise_and(msk,cv2.bitwise_not(GT))
                image = cv2.bitwise_and(image,image,mask=msk)
                
            if aug_prob == 3: #marker augmentation
                mms = np.load('mm_array.npy')  
                image = mm_aug(mms,image,GT)
                
                    
            Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            image = Norm_(image)  
            
            image = np.array(image).transpose(2, 0, 1)
            data = torch.tensor(image).float()
        '''
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        # Augmenting duplicated images to treat as new data points
        if self.transform2 is not None:
            if self.csv.marked[index] == 1:
                res = self.transform2(image=image)
                image = res['image'].astype(np.float32)
        
        if self.mode == 'train':
            aug_prob = 1
    
            if aug_prob == 1: #hair augmentation
                hairs = np.load('hair_array.npy')
                msk = hair_mask(hairs,IMAGE_SIZE=256).astype(np.uint8)
                image = cv2.bitwise_and(image,image,mask= msk)   
                
        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        if self.mode == 'test':
            return data, torch.tensor(self.csv.iloc[index].target).long()

        # Returning different data based on what test is being run
        else:                
            return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                self.csv.iloc[index].marked).long(), torch.tensor(self.csv.iloc[index].scale).long()

        #return data, torch.tensor(self.csv.iloc[index].target).long

    def __len__(self):
        return self.csv.shape[0]

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader


''' #################################################################################################################### '''
# Augmentations
def get_transforms():
    # Augmentations for oversampled images to treat them as new data points
    transforms_marked = None

    # Augmentations for all training data
    
    transforms_train = albu.Compose([
        albu.Resize(args.image_size, args.image_size),
        albu.Normalize()
    ])

    # Augmentations for validation data
    transforms_val = albu.Compose([
        albu.Resize(args.image_size, args.image_size),
        albu.Normalize()
    ])
    return transforms_marked, transforms_train, transforms_val

def get_df():
    # Loading test csvs
    df_test_blank = pd.read_csv(os.path.join(args.csv_dir, 'holger_blank.csv'))
    # Adding column with path to file
    
    df_test_atlasD = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    df_test_atlasD['filepath'] = df_test_atlasD['derm'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')
    
    df_test_atlasC = pd.read_csv(os.path.join(args.csv_dir, 'atlas_processed.csv'))
    df_test_atlasC['filepath'] = df_test_atlasC['clinic'].apply(
        lambda x: f'{args.image_dir}/atlas_{args.image_size}/{x}')
    
    df_test_ASAN = pd.read_csv(os.path.join(args.csv_dir, 'asan.csv'))
    df_test_ASAN['filepath'] = df_test_ASAN['image_name'].apply(
        lambda x: f'{args.image_dir}/asan_{args.image_size}/{x}')
    
    df_test_MClassD = pd.read_csv(os.path.join(args.csv_dir, 'MClassD.csv'))
    df_test_MClassD['filepath'] = df_test_MClassD['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassD_{args.image_size}/{x}')
    
    df_test_MClassC = pd.read_csv(os.path.join(args.csv_dir, 'MClassC.csv'))
    df_test_MClassC['filepath'] = df_test_MClassC['image_name'].apply(
        lambda x: f'{args.image_dir}/MClassC_{args.image_size}/{x}')
    
    # Placeholders for dataframes that are conditionally instantiated
    df_val = []
    
    # Loading train csv
    df_train = pd.read_csv(os.path.join(args.csv_dir, 'isic_train_20-19-18-17.csv'), low_memory=False)

    if args.marked and args.skew:
        # Removing benign marked images and duplicating malignant marked to skew data
        df_train_benign_marked = df_train.loc[(df_train.marked == 1) & (df_train.benign_malignant == 'benign'), :]
        df_train['remove'] = 0
        df_train.loc[df_train.image_name.isin(df_train_benign_marked.image_name), 'remove'] = 1
        df_train = df_train[df_train.remove != 1]

        marked_df = df_train.loc[df_train.marked == 1, :]

        for i in range(args.duplications_m):
            df_train = pd.concat([marked_df, df_train])

    if args.rulers and args.skew:
        # Removing benign ruler images and duplicating malignant ruler to skew data
        df_train_benign_scale = df_train.loc[(df_train.scale == 1) & (df_train.benign_malignant == 'benign'), :]
        df_train.loc[df_train.image_name.isin(df_train_benign_scale.image_name), 'remove'] = 1
        df_train = df_train[df_train.remove != 1]
        scale_df = df_train.loc[df_train.scale == 1, :]
        for i in range(args.duplications_r):
            df_train = pd.concat([scale_df, df_train])

    # Removing overlapping Mclass images from training data to prevent leakage
    df_train = df_train.loc[df_train.mclassd != 1, :]
    
    # Removing 2020 Data from training set.
    df_train = df_train.loc[df_train.year != 2020,:]
    
    # Removing 2019 comp data from training data
    df_train = df_train.loc[df_train.year != 2019, :]
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)

    # Setting cv folds for 2017 data
    df_train.loc[(df_train.year != 2020) & (df_train.year != 2018), 'fold'] = df_train['tfrecord'] % 5
    tfrecord2fold = {
        2: 0, 4: 0, 5: 0,
        1: 1, 10: 1, 13: 1,
        0: 2, 9: 2, 12: 2,
        3: 3, 8: 3, 11: 3,
        6: 4, 7: 4, 14: 4,
    }
    # Setting cv folds for 2020 data
    df_train.loc[(df_train.year == 2020), 'fold'] = df_train['tfrecord'].map(tfrecord2fold)
    # Putting image filepath into column
    df_train.loc[(df_train.year == 2020), 'filepath'] = df_train['image_name'].apply(
        lambda x: os.path.join(f'{args.image_dir}/isic_20_train_{args.image_size}/{x}.jpg'))
    df_train.loc[(df_train.year != 2020), 'filepath'] = df_train['image_name'].apply(
        lambda x: os.path.join(f'{args.image_dir}/isic_19_train_{args.image_size}', f'{x}.jpg'))

    # Get validation set for hyperparameter tuning
    df_val = df_train.loc[df_train.year == 2018, :].reset_index()
    df_val['instrument'] = 0  # Adding instrument placeholder to prevent error
    _, df_val = train_test_split(df_val, test_size=0.33, random_state=args.seed, shuffle=True)
    # Removing val data from training set
    df_train = df_train.loc[df_train.year != 2018, :]

    if args.instrument:
        # Keeping only most populated groups of image sizes to use as proxy for instruments
        keep = ['6000x6000', '1872x1872', '640x640', '5184x5184', '1024x1024',
                '3264x3264', '4288x4288', '2592x2592']
        df_train = df_train.loc[df_train['size'].isin(keep), :]
        # mapping image size to index as proxy for instrument
        size2idx = {d: idx for idx, d in enumerate(sorted(df_train['size'].unique()))}
        df_train['instrument'] = df_train['size'].map(size2idx)

    mel_idx = 1  # Setting index for positive class

    # Returning training, validation and test datasets
    return df_train, df_val, df_test_atlasD, df_test_atlasC,\
        df_test_ASAN, df_test_MClassD, df_test_MClassC, mel_idx