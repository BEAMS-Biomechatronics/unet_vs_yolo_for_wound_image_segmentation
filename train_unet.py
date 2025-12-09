import numpy as np
import os
import cv2
from PIL import Image
import math
import tensorflow as tf
from tensorflow import keras
import unet_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
from pathlib import Path

def calculate_metrics(mask1, mask2):
    assert mask1.shape == mask2.shape, "Les masques doivent avoir les mêmes dimensions"
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    iou = np.sum(intersection) / np.sum(union)
    
    true_positive = np.sum(intersection)
    false_positive = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    false_negative = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    dice_coefficient = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    
    return iou, precision, recall, dice_coefficient

class DataGen(keras.utils.Sequence):

    def __init__(self, path, batch_size, image_size, labels=True, shuffle=True, file_list=None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.path = path
        self.list_IDs = os.listdir(os.path.join(self.path, 'images'))
        if file_list is not None:
            self.list_IDs = file_list
        else:
            self.list_IDs = os.listdir(os.path.join(self.path, 'images'))
        if labels:
            self.label_IDs = os.listdir(os.path.join(self.path, 'masks'))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, 'images', id_name)
        mask_path = os.path.join(self.path, 'masks', id_name)

        ## Reading Image
        image = cv2.imread(image_path, 1)[:,:,::-1]
        image = cv2.resize(image, self.image_size)
        mask = cv2.imread(mask_path, 1)[:,:,::-1]
        mask = cv2.resize(mask[:,:,1], self.image_size)
        mask = np.expand_dims(mask, axis=-1)

        ## Normalizaing
        image = image/255.0
        mask = mask/255.0

        return image, mask

    def __getitem__(self, index):

        if(index+1)*self.batch_size > len(self.list_IDs):
            self.batch_size = len(self.list_IDs) - index*self.batch_size

        files_batch = self.list_IDs[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []
        for id_name in files_batch:
          _img, _mask = self.__load__(id_name)
          image.append(_img)
          mask.append(_mask)

        image = np.array(image)
        mask  = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)


def train_folds(base_path):
    print(base_path)
    model = unet_model.UNet(image_size=640)
    base_path = Path(base_path)
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=[BinaryAccuracy(), MeanIoU(num_classes=2)])

    #for subfolder in base_path.iterdir()
    subfolder = base_path
    if subfolder.is_dir():
        for i in range(1, 6):  # Itérer de fold1 à fold5
            fold_path = subfolder / f'fold_{i}'

            train_gen = DataGen(path=fold_path / 'train', batch_size=16, image_size=(640,640))
            val_gen = DataGen(path=fold_path / 'val', batch_size=16, image_size=(640,640))
            
            model.fit(train_gen, epochs=100, validation_data=val_gen)
            
            model.save_weights(f'unet_weights_combined_AZHtest_fold_{i}.h5')

            print(f"Completed training for {subfolder.name}/fold_{i}")

    print(f"Completed training for all models in {subfolder.name}")


base_path = Path("C:\\Users\\Haroun\\Desktop\\unet")  #Path(".\\yamls\\")
train_folds(base_path)
