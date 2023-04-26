import os
import librosa
import glob
import numpy as np
import itertools
from matplotlib import pyplot as plt
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
if sklearn.__version__ > '0.18':
    from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers, Sequential, Input

import pickle

tf.random.set_seed(1)

reSampledFreq = 8000
_row = 128,
_col = 49


def usePickle(xTest):

    print("Loading Predict function from pickle\n")
    loaded_model = pickle.load(open('model.pkl','rb'))
    score_list = loaded_model.predict(xTest.reshape(1,128,79,1))
    print("\n",score_list)
    # print("\nusePickle Result: ", round(score_list,2), "\n")
    result = "No Cough" if np.argmax(score_list) == 0 else "Cough"

    return result



def predict_xTest(xTest_path):

    print("Loading audio...")
    print("File Path: ",xTest_path,"\n")

    xTest, fs_new= preProcessData(xTest_path, reSampledFreq)

    # print("Downsampled frequency is: ",fs_new," Hz")
    print("\nxTest.shape: ",xTest.shape,"\n")

    spec = librosa.feature.melspectrogram(y=xTest, sr=fs_new)

    result = usePickle(spec)
    print("\nResult: ",result,"\n")

    return result



def fitModel(model, X, Y):

    # tf.random.set_seed(1)

    print("Fitting model...\n")

    X1, X2, y1, y2 = train_test_split(X, Y, random_state=0,test_size=0.2)
    number_of_epochs = 10
    batch_size = 50

    print('# Fit model on training data')
    model.fit(X1, y1, batch_size = batch_size, epochs = number_of_epochs, validation_data = (X2,y2))

    pickle.dump(model, open('model.pkl','wb'))

    return model



def compileModel():

    print("Compiling model...")
    model = Sequential()
    model.add(Input(shape=(128,79,1)))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=32,kernel_size=(3,2),padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(2,activation = 'softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    model.summary()

    return model



def trainModel(X,Y):

    skeletonModel = compileModel()
    model = fitModel(skeletonModel, X, Y)

    return model



def create_Y(listOfAllSoundFiles):

    print("Creating Y...")

    classArray = np.zeros((len(listOfAllSoundFiles),1)) #array to store classes

    counterCough = 0

    #creating classes labels in the classArray......
    for i in range(len(listOfAllSoundFiles)):
        if listOfAllSoundFiles[i].split('/')[-2] == 'cough':
            classArray[i] = 1
            counterCough+=1
        else:
            classArray[i] = 0

    print("\ncounterCough: ",counterCough)
    print("Total Samples:", len(listOfAllSoundFiles),"\n")

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(classArray)
    Y = enc.transform(classArray).toarray()

    print("Y.shape: ",Y.shape)

    return Y



def preProcessData(audioFile, cutoff):

    s,sr = librosa.load(audioFile, sr=None, duration=5.0)
    resampled_sr = librosa.resample(y=s, orig_sr=sr, target_sr=cutoff)
    normalized_data = librosa.util.normalize(resampled_sr, axis=0)
    # print("fileIndex:",fileIndex,"\nog_s.shape: ",s.shape,"\nnr_s.shape: ",normalized_data.shape)

    return normalized_data, cutoff



def create_X(listOfAllSoundFiles):

    print("\nCreating X...")

    X = np.zeros((len(listOfAllSoundFiles),128,79)) #array to store features

    specShapeList = []

    start = datetime.now()

    for fileIndex in range(len(listOfAllSoundFiles)):
        oneFileFeature, fs_new = preProcessData(listOfAllSoundFiles[fileIndex], 8000)
        spec = librosa.feature.melspectrogram(y=oneFileFeature, sr=fs_new)
        if spec.shape == (128,79):
            X[fileIndex] = spec
            continue
        else:
            print("Non standard spec space at index: ", fileIndex)
            otherShape = str('{} : {}').format(fileIndex,spec.shape)
            specShapeList.append(otherShape)

            # if os.path.isfile(listOfAllSoundFiles[fileIndex]):
            #     os.remove(listOfAllSoundFiles[fileIndex])
    
    print("X.shape: ",X.shape)
    duration = datetime.now() - start
    print("spec_shape: ",specShapeList)
    print("\nFeatures extracted in time: {} \n".format(duration))

    return X



def createInputs():

    # trainDataWebRepoPath = "https://1drv.ms/f/s!AjWVWtWhmYVShPMiU8xnsqtAcgbYXQ"

    print("Creating soundfile_list...")

    trainData_path = "/Users/madhavdatt/CSE5349/trainData"
    soundfiles=[]
    # trainingDataDirectory = './audio2'
    dirList = os.listdir(trainData_path)


    # Globbing all the sound files........
    for dir in dirList:
        currentDirFile = glob.glob(trainData_path+'/'+dir+'/'+'*.wav')
        soundfiles.append(currentDirFile)


    # Flattenning soundfiles list into a single list...........
    listOfAllSoundFiles=list(itertools.chain(*soundfiles))
    # random.shuffle(listOfAllSoundFiles)

    # print("Total Samples: ",len(listOfAllSoundFiles),"\n")

    # print("listOfAllSoundFiles: ",listOfAllSoundFiles)

    # shortAudioSampleIndex_List = []

    # for fileIndex in range(len(listOfAllSoundFiles)):
    #     # print("File: ",listOfAllSoundFiles[i])
    #     if librosa.get_duration(path=listOfAllSoundFiles[fileIndex]) < 5.0:
    #         shortAudioSampleIndex_List.append(fileIndex)
            
    # print(shortAudioSampleIndex_List)

    # for fileIndex in shortAudioSampleIndex_List:
    #     os.remove(listOfAllSoundFiles[fileIndex])

    # s = pd.Series(wavFile_Duration)
    # print(s.describe())

    # listOfAllSoundFiles = listOfAllSoundFiles[:100]

    # soundfile_list = soundfile_list[:100]

    print("Total samples: ",len(listOfAllSoundFiles))

    X = create_X(listOfAllSoundFiles)
    Y = create_Y(listOfAllSoundFiles)


    # soundfile_list = glob.glob(trainData_path+'/'+'*.wav')
    # random.shuffle(soundfile_list)

    # soundfile_list = soundfile_list[:100]

    # print(soundfile_list)

    # soundfile_list = soundfile_list[:5]


    # for i in range(len(soundfile_list)):
    #     print(soundfile_list[i][-6],soundfile_list[i][-5]," ",X[i,-1,-1]," ",Y[i])

    return X,Y



def cdClassifier():

    print("\nCoughDetectionModel.py is executing...\n")

    X,Y = createInputs()

    trainModel(X,Y)

    xTest_path = "./1-53663-A-24.wav"
    # xTest2_path = "./cough_recorded_04242023.wav"
    xTest3_path = "./cd_sr.wav"
    xTest4_path = "./cd_mt.wav"

    predict_xTest(xTest_path)
    # predict_xTest(xTest2_path)
    predict_xTest(xTest3_path)
    predict_xTest(xTest4_path)

    # print(*soundfile_list,sep='\n')
    # print('\nhistory dict:', history.history)