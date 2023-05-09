import tensorflow as tf
import keras
from keras import Sequential
from keras.preprocessing import image
from keras.utils import load_img,img_to_array

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

##Data preprocessing
#Read the data file
df = pd.read_csv('train.csv')
#Check null value
df.isnull().sum()
#Get 6000 result for testing/validating
df = df.head(6000)
#Remove duplicate rows. 
df.drop_duplicates(subset=['Id'], keep="first")
#Remove minority classes
df = df.drop(['NA','News','Reality-TV','Short','Western'],axis=1)

##Prepare train data and test data
#Set width/height of images to 350
#Append images to numpy array
width = 350
height = 350
X = []
for i in tqdm(range(df.shape[0])):
  path = 'Images/'+df['Id'][i]+'.jpg'
  img = load_img(path,target_size=(width,height,3))
  img = img_to_array(img)
  img = img/255.0
  X.append(img)

X = np.array(X)
img_flat = X.reshape(len(X),-1)

#Reshape X to 2D array (number of images,dimension*size)
X = X.reshape(len(X),3*350*350)

#Drop the first 2 columns and convert the rest to numpy array
y = df.drop(['Id','Genre'],axis=1)
y = y.to_numpy()

#Set the train/test to 7:3 and random state to 10
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,
                                                random_state=10)


#Generate kNN model
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data_normalized = scaler.fit_transform(X_train)
model = KNeighborsClassifier(n_neighbors=5)


#Fit model to the dataset
#model.fit(data_normalized, y_train)
model.fit(X_train, y_train)


#Accuracy and misclassification result
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print('Accuracy:', accuracy)
misclassification_error = 1 - np.mean(predictions == y_test)
print('Misclassification error:', misclassification_error)


#Test model
img = load_img('Images/tt2505856.jpg',target_size=(width,height,3))
plt.imshow(img)
img = img_to_array(img)
img = img/255.0
img = img.reshape(1,3*350*350)
classes = df.columns[2:]
y_pred = model.predict(img)
top3=np.argsort(y_pred[0])[:-4:-1]
for i in range(3):
  print(classes[top3[i]])

