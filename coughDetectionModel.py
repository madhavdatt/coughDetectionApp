
import librosa
import scipy.signal as signal
import os
import glob
import random
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
if sklearn.__version__ > '0.18':
    from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers, Sequential, Input

import traceback
import time
import pickle


# def predict(model,xTest):

#     print("\nLoading Predict function from 'model'\n")
#     pred_list = model.predict(xTest.reshape(1,128,157,1))
#     result = "No Cough" if np.argmax(pred_list) == 0 else "Cough"
#     print("Result: ",result)



def usePickle(xTest):

    print("Loading Predict function from pickle\n")
    loaded_model = pickle.load(open('model.pkl','rb'))
    pred_list = loaded_model.predict(xTest.reshape(1,128,157,1))
    print("\nusePickle Result: ", pred_list, "\n")
    result = "No Cough" if np.argmax(pred_list) == 0 else "Cough"

    return result



def predict_xTest(xTest_path):

    print("\nLoading audio from path",xTest_path,"\n")

    xTest, fs_new= preProcessData(xTest_path, 8000, True, True)
    print("Downsampled frequency is: ",fs_new," Hz")
    print("\nxTest.shape: ",xTest.shape,"\n")

    spec = librosa.feature.melspectrogram(y=xTest, sr=fs_new)

    result = usePickle(spec)
    print("\nResult: ",result,"\n")

    return result



def fitModel(model, X, Y):
    
    print("Fitting model...\n")

    X1, X2, y1, y2 = train_test_split(X, Y, random_state=0,test_size=0.1)
    number_of_epochs = 5
    batch_size = 18

    print('# Fit model on training data')
    model.fit(X1, y1, batch_size = batch_size, epochs = number_of_epochs, validation_data = (X2,y2))

    pickle.dump(model, open('model.pkl','wb'))

    return model



def compileModel():

    print("Compiling model...")
    model = Sequential()
    model.add(Input(shape=(128,157,1)))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(2,activation = 'softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    return model



def trainModel(X,Y):

    skeletonModel = compileModel()
    model = fitModel(skeletonModel, X, Y)

    return model



def create_Y(soundfile_list):
    print("Creating Y...")
    classArray = np.zeros((len(soundfile_list),1)) #array to store classes

    counterCough = 0
    #creating classes labels in the classArray......
    for i in range(len(soundfile_list)):
        if soundfile_list[i][-6] == '2' and soundfile_list[i][-5] == '4':
            classArray[i] = 1
            counterCough+=1
        else:
            classArray[i] = 0

    print("\ncounterCough: ",counterCough)
    print("Total Samples:", len(soundfile_list),"\n")
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(classArray)
    Y = enc.transform(classArray).toarray()
    
    return Y



def preProcessData(audioFile, cutoff, normalize, downsample):

    origData,origSampFreq = librosa.load(audioFile, sr=None, duration=5.0)

    x = origData
    fs = origSampFreq
    fs_new = cutoff*2

    # print("Before downsampling: ",x.shape)
    
    if downsample:
        x = librosa.resample(y=x, orig_sr=fs, target_sr=fs_new)

    return x, fs_new



def create_X(soundfile_list):

    print("\nCreating X...")

    X = np.zeros((len(soundfile_list),128,157)) #array to store features

    for num in range(len(soundfile_list)):
        oneFileFeature, fs_new = preProcessData(soundfile_list[num], 8000, True, True)
        spec = librosa.feature.melspectrogram(y=oneFileFeature, sr=fs_new)
        # print(mfcc.shape)
        X[num] = spec

    return X



def create_X_Y():

    print("Creating soundfile_list...")
    trainingDataDirectory = '/Users/madhavdatt/CSE5349/audio2'
    dirList = os.listdir(trainingDataDirectory)

    soundfile_list=[]

    soundfile_list = glob.glob(trainingDataDirectory+'/'+'*.wav')
    random.shuffle(soundfile_list)

    # soundfile_list = soundfile_list[:5]

    X = create_X(soundfile_list)
    Y = create_Y(soundfile_list)

    return X,Y



def cdClassifier():

    tf.random.set_seed(0)

    print("\nCoughDetectionModel.py is executing...\n")

    X,Y = create_X_Y()
    trainModel(X,Y)

    xTest_path = "/Users/madhavdatt/CSE5349/coughDetectionApp/stockCoughSound.wav"
    predict_xTest(xTest_path)

    # print(*soundfile_list,sep='\n')
    # print('\nhistory dict:', history.history)