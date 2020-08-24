import segmentation_models as sm
import glob
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from keras.callbacks import EarlyStopping
import os

os.chdir(r'/scale_wlg_persistent/filesets/home/rampaln/SuperResolutionSatellite')
df = xr.open_dataset('Galaxy_Training_Dataset.nc')
model1 = sm.Unet(backbone_name='resnet18', activation = 'relu',
 		 encoder_freeze=True, input_shape = (64,64, 3), classes =3,
 		 weights = 'test_model_final.h5')


# Adding on to the model  
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(model1.output)
x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
x3 = BatchNormalization()(x3)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = UpSampling2D((2, 2))(x2)
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
#x1 = UpSampling2D((2, 2))(x1)
decoded   = Conv2D(64, (3, 3), padding='same', activation = 'relu')(x1)
decoded   = Conv2D(3, (3, 3), padding='same', activation = 'relu')(decoded)
autoencoder = Model(model1.input, decoded)

# compiling the model 
autoencoder.compile(optimizer='adam', loss='mse')


x_train = df['x_train'].values
x_test = df['x_test'].values
y_train = df['y_train'].values
y_test = df['y_test'].values
autoencoder.fit(x=x_train[:]/255.0, y=y_train[:,:,:]/255.0, validation_data=(x_test[:]/255.0, y_test[:,:,:]/255.0), batch_size =10, epochs =20)
autoencoder.save('output_encoder.h5')
x1 = autoencoder.predict(x_test[0:20]/255.0)

for in range(10):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(x1[i])
    ax[1].imshow(x_test[i])
    ax[2].imshow(y_test[i])
    fig.savefig(f'test_img_{i}.pdf',dpi=100)