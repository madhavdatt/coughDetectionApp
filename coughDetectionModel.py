
import librosa
import os
import glob
import random
import numpy as np
from librosa.util import normalize
from sklearn.preprocessing import OneHotEncoder
import sklearn
if sklearn.__version__ > '0.18':
    from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers, Sequential, Input
import pickle


def predict2_xTest(xTest_path):
    print("\Loading recorded audio...")
    origData, origSampFreq = librosa.load(xTest_path, sr=8000, duration=5.0)
    normalizedData = normalize(origData, axis=0)
    xTest=librosa.feature.melspectrogram(y=normalizedData, sr=8000, hop_length=256, win_length=256)
    print("\nTest.shape: ",xTest.shape,"\n")
    print("For path:",xTest_path)
    result = usePickle(xTest)
    return result


def predict(model,xTest):
    print("\nPredict:")
    pred_list = model.predict(xTest.reshape(1,128,157,1))
    result = "No Cough" if np.argmax(pred_list) == 0 else "Cough"
    print("Result: ",result)


def usePickle(xTest):
    print("\nLoading Pickle File...")
    loaded_model = pickle.load(open('model.pkl','rb'))
    pred_list = loaded_model.predict(xTest.reshape(1,128,157,1))
    print("\nusePickle Result: ", pred_list, "\n")
    result = "No Cough" if np.argmax(pred_list) == 0 else "Cough"
    return result


def buildPickle(model):
    pickle.dump(model, open('model.pkl','wb'))


def predict_xTest(model, xTest_path):
    print("\nLoading stock audio...")
    origData, origSampFreq = librosa.load(xTest_path, sr=8000, duration=5.0)
    normalizedData = normalize(origData, axis=0)
    xTest=librosa.feature.melspectrogram(y=normalizedData, sr=8000, hop_length=256, win_length=256)
    print("\nTest.shape: ",xTest.shape,"\n")
    print("For path:",xTest_path)
    predict(model,xTest)
    buildPickle(model)


def fitModel(model, X, Y):
    print("Fitting model...")

    X1, X2, y1, y2 = train_test_split(X, Y, random_state=0,test_size=0.1)
    number_of_epochs = 3
    batch_size = 20

    print('# Fit model on training data')
    history = model.fit(X1, y1, batch_size = batch_size, epochs = number_of_epochs, validation_data = (X2,y2), verbose=1)
    return model, history


def compileModel():
    print("Compiling model...")
    model = Sequential()
    model.add(Input(shape=(128,157,1)))
    model.add(layers.Conv2D(filters=32,kernel_size=(2,2),padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=32,kernel_size=(2,2),padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(2,activation = 'softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    return model


def trainModel(X,Y):
    skeletonModel = compileModel()
    model, history = fitModel(skeletonModel, X, Y)
    return model, history


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

    print("counterCough: ",counterCough)
    print("Total Samples:", len(soundfile_list))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(classArray)
    y_train = enc.transform(classArray).toarray()
    
    return y_train


def readCoughData(file):
    origData,origSampFreq = librosa.load(file, sr=8000, duration=5.0)
    return origData, origSampFreq


def normalizeSound(origData, axis):
    normalizedData = normalize(origData, axis=axis)
    return normalizedData


def calculateMelSpectogram(normalizedData, hop_length, win_length, sr):
    S=librosa.feature.melspectrogram(y=normalizedData, sr=sr, hop_length=hop_length, win_length=win_length)
    return S


def featureExtraction(audioFile, targetSampFreq, axis, hop_length,win_length):
    y, y_sr = readCoughData(file=audioFile)
    normalizedData = normalizeSound(y, axis=axis)
    S = calculateMelSpectogram(normalizedData=normalizedData, hop_length=hop_length, win_length=win_length, sr=targetSampFreq)
    return S


def create_X(soundfile_list):
    print("Creating X...")
    featureArray = np.zeros((len(soundfile_list),128,157)) #array to store features

    # Extracting features for all the 2000 files and collecting them in featureArray
    for num in range(len(soundfile_list)):
        featureArray[num] = featureExtraction(soundfile_list[num], targetSampFreq=8000, axis=0, hop_length=256, win_length=256)
    return featureArray


def create_soundfile_list():
    print("Creating soundfile_list...")
    trainingDataDirectory = '/Users/madhavdatt/CSE5349/audio2'
    dirList = os.listdir(trainingDataDirectory)

    soundfile_list=[]

    for dir in dirList:
        soundfile_list = glob.glob(trainingDataDirectory+'/'+'*.wav')
    random.shuffle(soundfile_list)

    return soundfile_list


def runCD_Train():
    tf.random.set_seed(5368)
    soundfile_list = create_soundfile_list()

    X = create_X(soundfile_list)
    Y = create_Y(soundfile_list)
    
    model, history = trainModel(X,Y)
    print('\nhistory dict:', history.history)
    return model


# print(*soundfile_list,sep='\n')
# soundfile_list = soundfile_list[:100]

model = runCD_Train()

xTest_path = "/Users/madhavdatt/CSE5349/coughDetectionApp/stockCoughSound.wav"
predict_xTest(model, xTest_path)