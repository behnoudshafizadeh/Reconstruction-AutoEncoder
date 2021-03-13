# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import cv2
import os
np.random.seed(42)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# initialize the number of epochs to train for and batch size
EPOCHS = 500
BS = 32

#make correct train
SIZE=28
path1='/home/user1/custom-autoencoder/clean_train'
sorted_list_clean=[]
finall_list_clean=[]
clean_data=[]

for f in os.listdir(path1):
    file_name,file_ext=os.path.splitext(f)
    sorted_list_clean.append(file_name)
sorted_list_clean.sort()

for f in sorted_list_clean:
    f=f+'.png'
    finall_list_clean.append(f)

for i in finall_list_clean:
    img=cv2.imread(path1+"/"+i,0)
    try:
        img=cv2.resize(img,(SIZE,SIZE))
    except cv2.error as e:
        print('Invalid frame!')
    clean_data.append(img)    


#make distort data
path2='/home/user1/custom-autoencoder/noisy_train'
sorted_list_noisy=[]
finall_list_noisy=[]
noisy_data=[]

for f in os.listdir(path2):
    file_name,file_ext=os.path.splitext(f)
    sorted_list_noisy.append(file_name)
sorted_list_noisy.sort()

for f in sorted_list_noisy:
    f=f+'.png'
    finall_list_noisy.append(f)

for i in finall_list_noisy:
    img=cv2.imread(path2+"/"+i,0)
    try:
        img=cv2.resize(img,(SIZE,SIZE))
    except cv2.error as e:
        print('Invalid frame!')
    noisy_data.append(img)


#reshape data to batch
noisy_train=np.reshape(noisy_data,(len(noisy_data),SIZE,SIZE,1))
noisy_train=noisy_train.astype('float32')/255.0
clean_train=np.reshape(clean_data,(len(clean_data),SIZE,SIZE,1))
clean_train=clean_train.astype('float32')/255.0


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)


#split data to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(noisy_train, clean_train,test_size=0.1,random_state = 0)

#fit model
H=autoencoder.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=EPOCHS,batch_size=BS)


# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("paper-result-run5.png")

autoencoder.save("weights-of-run5.h5")
