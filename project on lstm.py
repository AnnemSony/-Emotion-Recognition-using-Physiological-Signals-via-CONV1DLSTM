
from tabnanny import verbose
from turtle import shape
from matplotlib.axis import Axis
from matplotlib.cbook import flatten
from pyrsistent import v
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#import tensorflow_datasets as tfds
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import sys
import os
import fnmatch
from keras.models import load_model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.layers import LSTM
from sklearn import preprocessing
#timesteps = 12
timesteps = 1000
batc_size = 1000
input_dim = 100 #number of fetures
#voc_size = (input_dim.max()+1)
def GetData(trainortest,datapath,modelname,datatype):
    #print(modelname)
    # Get the list of all files and directories
   # path = "C:/Users/annem/OneDrive/Desktop/project2/Physiological/Training"
    trainingData = []
    validationData = []
    testingData = []
    trainingLabels = []
    validationLabels = []
    testingLabels = []
    if trainortest == "train":
        path = datapath + "/Training" 
        print("Files and directories in '", path, "' :")
        dir_list = os.listdir(path)
        #print(len(dir_list))
        for file_path in dir_list:
            file1 = open(path + "\\" +file_path,'r')
            label=file_path.split("_")   
            file2=label[2]+"_"+label[3]
            if datatype=="EDA":
               if "EDA_microsiemens.txt" != file2:
                   continue
            if datatype=="mmhg":
               if "BP_mmHg.txt" != file2:
                   continue
            if datatype=="mean":
               if "LA Mean BP_mmHg.txt" != file2:
                   continue
            if datatype=="sys":
               if "LA Systolic BP_mmHg.txt" != file2:
                   continue
            if datatype=="pulse":
               if "Pulse Rate_BPM.txt" != file2:
                   continue
            if datatype=="DIA":
               if "BP Dia_mmHg.txt" != file2:
                   continue
            if datatype=="volt":
               if "Resp_Volts.txt" != file2:
                   continue
            if datatype=="resp":
               if "Respiration Rate_BPM.txt" != file2:
                   continue   
            #print(label[1])        
            trainingLabels.append([int(label[1])-1])
            if int(label[1])-1==10:
                print(file_path)
            #print(int(label[1])-1)
            #scaler = MinMaxScaler(feature_range=(-1, 1))
            #trainingData =scaler.fit(trainingData)
            Lines = file1.readlines()
            #print(Lines)
            count = 0
            filelist=[]
            # # Strips the newline character
            for line in Lines:
                count += 1
                filelist.append(float(line.strip()))
                #print(line.strip())
            trainingData.append(filelist)
            #print(type(trainingData))
            
          
            #trainingData.remove(val)
            #print(type(trainingLabels))
                #trainingData=np.asarray(trainingData)
        path = datapath + "/Validation" 
        print("Files and directories in '", path, "' :")
        dir_list = os.listdir(path)
        #print(len(dir_list))
        for file_path in dir_list:
            #print(file_path)
            file1 = open(path + "\\" +file_path,'r')
            label=file_path.split("_")   
            file2=label[2]+"_"+label[3]
            #print(file2)
            if datatype=="EDA":
               if "EDA_microsiemens.txt"!= file2:
                   continue
            if datatype=="mmhg":
               if "BP_mmHg.txt"!= file2:
                   continue
            if datatype=="mean":
               if "LA Mean BP_mmHg.txt" != file2:
                   continue
            if datatype=="sys":
               if "LA Systolic BP_mmHg.txt" != file2:
                   continue
            if datatype=="pulse":
               if "Pulse Rate_BPM.txt" != file2:
                   continue
            if datatype=="DIA":
               if "BP Dia_mmHg.txt" != file2:
                   continue
            if datatype=="volt":
               if "Resp_Volts.txt" != file2:
                   continue
            if datatype=="resp":
               if "Respiration Rate_BPM.txt" != file2:
                   continue   
            # if datatype=="all":
            #      continue
            #print(label[1])        
            validationLabels.append([int(label[1])-1])
            #print(int(label[1])-1)
            Lines = file1.readlines()
            count = 0
            filelist=[]
            # # Strips the newline character
            for line in Lines:
                count += 1
                #print(line.strip())
                filelist.append(float(line.strip()))
                #print(line.strip())
            validationData.append(filelist)  
           
                   
        #trainingData=tf.keras.preprocessing.sequence.pad_sequences(trainingData, maxlen=None)
        #validationData=tf.keras.preprocessing.sequence.pad_sequences(validationData,  maxlen=None)
        trainingData=tf.keras.preprocessing.sequence.pad_sequences(trainingData, maxlen=timesteps)
        validationData=tf.keras.preprocessing.sequence.pad_sequences(validationData, maxlen=timesteps)
        trainingData=np.array(trainingData)
        trainingLabels=np.array(trainingLabels)
        #print(trainingLabels) 
        #print(trainingData)
       # print(trainingLabels.shape)
       
        validationData=np.array(validationData)
        validationLabels=np.array(validationLabels)
        trainingData=preprocessing.normalize(trainingData)
        validationData=preprocessing.normalize(validationData)
        trainingData= np.expand_dims(trainingData, axis=2)
        validationData= np.expand_dims(validationData, axis=2)
         

        #print(trainingData.shape)
        return trainingData, trainingLabels, validationData, validationLabels 
    elif trainortest == "test":
        path = datapath + "/Testing" 
        print("Files and directories in '", path, "' :")
        dir_list = os.listdir(path)
        #print(len(dir_list))
        for file_path in dir_list:
            file1 = open(path + "\\" +file_path,'r')
            label=file_path.split("_")   
            file2=label[2]+"_"+label[3]
            if datatype=="EDA":
               if "EDA_microsiemens.txt"!= file2:
                   continue
            if datatype=="mmhg":
               if "BP_mmHg.txt"!= file2:
                   continue
            if datatype=="mean":
               if "LA Mean BP_mmHg.txt"!= file2:
                   continue
            if datatype=="sys":
               if "LA Systolic BP_mmHg.txt"!= file2:
                   continue
            if datatype=="pulse":
               if "Pulse Rate_BPM.txt"!= file2:
                   continue
            if datatype=="DIA":
               if "BP Dia_mmHg.txt"!= file2:
                   continue
            if datatype=="volt":
               if "Resp_Volts.txt"!= file2:
                   continue
            if datatype=="resp":
               if "Respiration Rate_BPM.txt"!= file2:
                   continue   
            #print(label[1])        
            testingLabels.append([int(label[1])-1])
            Lines = file1.readlines()
            count = 0
            filelist=[]
            # # Strips the newline character
            for line in Lines:
                count += 1
                filelist.append(float(line.strip()))
                #print(line.strip())
            testingData.append(filelist)         
        testingData=tf.keras.preprocessing.sequence.pad_sequences(testingData, maxlen=timesteps)
        testingData=np.array(testingData)
        testingLabels=np.array(testingLabels)
        #tesData=preprocessing.normalize(trainingData)
        testingData=preprocessing.normalize(testingData)
        testingData= np.expand_dims(testingData, axis=2)
        
        #print(testingLabels) 
        return testingData, testingLabels, validationData, validationLabels

def MakeModel(): 
    model = tf.keras.models.Sequential()
   # model.add(tf.keras.layers.Embedding(batc_size, input_dim, input_length=(timesteps,1)))
    model.add(tf.keras.layers.Conv1D(64, kernel_size=3, input_shape=(timesteps,1), activation="relu"))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.InputLayer(input_shape=(timesteps,1)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    #model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
    #loss='categorical_crossentropy'loss='kullback_leibler_divergence', 'sparse_categorical_crossentropy'
   # model.summary() 
    return model  
def Predict(model, testingData, testingLabels):
    #predict and format output to use with sklearn
    predict = model.predict(testingData)
    predict = np.argmax(predict, axis=1)
    #macro precision and recall
    #metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    precisionMacro = precision_score(testingLabels, predict, average='macro', zero_division=1)
    recallMacro = recall_score(testingLabels, predict, average='macro', zero_division=1)
    #micro precision and recall
    precisionMicro = precision_score(testingLabels, predict, average='micro', zero_division=1)
    recallMicro = recall_score(testingLabels, predict, average='micro', zero_division=1)
    #try without zero_division
    confMat = confusion_matrix(testingLabels, predict)
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print(confMat)
def Train(trainortest,datapath,modelname,datatype):
    trainingData, trainingLabels, validationData, validationLabels  = GetData(trainortest,datapath,modelname,datatype)
    model = MakeModel()
    #verbose :shows progress of training speed

    model.fit(trainingData, trainingLabels,batch_size=input_dim,validation_data=[validationData, validationLabels],epochs=20)
        
    #model.fit(np.asarray(trainingData), np.asarray(trainingLabels), verbose=0, epochs=1)
    #model.fit(tf.expand_dims(trainingData,axis=-1),trainingLabels,epochs=1)
    #.h5 is extension
    model.save("./models/"+modelname+".h5")
    #model.input_shape()
    #model.summary() 
    print("Model saved.")

def Test(trainortest,datapath,modelname,datatype):
    print("Loading Test Data")
    #print(datapath)
    #print(modelname)
    testingData, testingLabels, trainingzData, trainingLabels = GetData(trainortest,datapath,modelname,datatype)
    print("Loading model")
    model = tf.keras.models.load_model("./models/"+modelname+".h5")
    print("Making predictions on test data")
    Predict(model, testingData, testingLabels)
    #Predict()   
def main():
    trainortest= sys.argv[1]
    datapath= sys.argv[2]
    modelname= sys.argv[3]
    datatype= sys.argv[4]
    if sys.argv[1]=="train":
         Train(trainortest,datapath,modelname,datatype) 
    if sys.argv[1]=="test":
         Test(trainortest,datapath,modelname,datatype)
   
    GetData(trainortest,datapath,modelname,datatype) 
       
if __name__ == "__main__":
    main()
    