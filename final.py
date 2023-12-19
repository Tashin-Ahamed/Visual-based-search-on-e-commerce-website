import os
# import cv2
#import sns
import numpy as np
from numpy.linalg import norm
# from tqdm import tqdm
import os
# import PIL
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
img_size =500
# import pandas as pd
model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')


batch_size = 128
root_dir = "/Users/tashinahamed/Desktop/image/"

img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

datagen = img_gen.flow_from_directory(root_dir,
                                        target_size=(img_size, img_size),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

num_images = len(datagen.filenames)
num_epochs = int(math.ceil(num_images / batch_size))

feature_list = model.predict_generator(datagen, num_epochs,verbose = 1)
print("Num images   = ", len(datagen.classes))
print("Shape of feature_list = ", feature_list.shape)
filenames = [root_dir + '/' + s for s in datagen.filenames]
neighbors = NearestNeighbors(n_neighbors=8,
                             algorithm='ball_tree',
                             metric='euclidean')
neighbors.fit(feature_list)
f=open("/Users/tashinahamed/Desktop/t.txt","r")
# img_path = "/Users/tashinahamed/Desktop/image/Tshirt/2.JPG"
ff=""

img_path = f.readline()
flag=0
for index in img_path:
    if index=='"':
        flag+=1
    if flag==1:
        if index!='"':
            ff+=index
    if flag==2:
        break
img_path=ff
print(img_path)


f.close()
input_shape = (img_size, img_size, 3)
img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
test_img_features = model.predict(preprocessed_img, batch_size=1)

_, indices = neighbors.kneighbors(test_img_features)
f=open("/Users/tashinahamed/Desktop/t2.txt","w")
f.truncate(0)
for index in indices[0]:
    f.write(filenames[index])
    f.write("\n")

f.close()               
def similar_images(indices):
    plt.figure(figsize=(15,10), facecolor='white')
    plotnumber = 1    
    for index in indices:
        if plotnumber<=len(indices) :
            ax = plt.subplot(2,4,plotnumber)
            # img=cv2.imread(filenames[index])
            # cv2.imshow("ff",img)
            plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')            
            plotnumber+=1
    plt.tight_layout()

# similar_images(indices[0])
# img=cv2.imread(filenames[0])
# cv2.imshow("ff",img)
# print(indices.shape)

# plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
# plt.xlabel(img_path.split('.')[0] + '_Original Image',fontsize=20)
# plt.show()
# print('********* Predictions ***********')

