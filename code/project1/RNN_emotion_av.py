import random

import numpy as np
from time import time
from keras.layers import GRU, Bidirectional, Dense, Dropout, AveragePooling1D, Flatten
from keras.models import Sequential, load_model
from keras.initializers import Constant
from matplotlib import pyplot as plt
# feature dimension, GRU stack depth, Dense, output classes
# from livelossplot import PlotLossesKeras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys
from MyDataGenerator import DataGenerator
from tqdm import *
import copy
import scipy.signal as signal  # pip install scipy==1.1.0
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
# from utils import plot_confuse

data_fixed_length = 300  # this is the fixed length of vectors in a character
number_of_classes = 3740  # this is the size of the shrunk dictionary.keys() size in PotIO class in IO.py
n_dimensions = 6

epochs = 30
batch_size = 512


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))

##net 6: 6-> [100,300,500] -> 100 -> 50
class RNN():
    train_set = []
    test_set = []
    test_labels = []
    train_labels = []
    conf_mat = None
    f1,precision,recall = 0,0,0
    type = '06'

    # dic['tag'] = [ Sample1: [stroke1: [TVs]]]

    def __init__(self,video_type):
        print("initiated RNN object, call ")
        self.type = video_type
        return

    def exec(self):
        global flag
        self.loadInternalRepresentationFiles() #从保存好的npy文件中读取训练集和测试集
        print(self.train_set.shape)
        print('transforming data')
        self.augumentDataSets() #数据预处理：补齐（多裁少补）
        print(self.train_set.shape)
        self.toNpArrs()
        print(self.train_set.shape)
        print(self.train_labels.shape)
        save_dir = "./all_data/"
        n_classes = len(self.k2l.keys())
        print("n_classes is {}".format(n_classes))
        self.save_files(save_dir, str(n_classes) + '-'+self.type)

        # n_classes = 50
        # self.load_features(n_classes)
        # RocAuc = RocAucEvaluation(validation_data=(self.train_set, self.train_labels), interval=1)

        self.model = self.buildRNN(n_classes)
        print('Starting training')
        filepath = 'E:/ocr_data/model/' + str(n_classes) + "-" + self.type + "-RNNmodel.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.history = self.model.fit(self.train_set, self.train_labels,
                                      validation_data=(self.test_set, self.test_labels),
                                      batch_size=batch_size, epochs=epochs, verbose=1,
                                      shuffle=True,callbacks=callbacks_list)

        # self.history = self.model.fit(self.train_set, self.train_labels,
        #                               validation_data=(self.test_set, self.test_labels),
        #                               batch_size=batch_size, epochs=epochs, verbose=1,
        #                               shuffle=True, callbacks=RocAuc)
        # self.model.save("RNNmodel.h5")

        # self.plotHistory()
        self.conf_mat,self.f1,self.precision,self.recall = self.plot_confuse(n_classes)
        print("self.f1,self.precision,self.recall is {},{},{}".format(self.f1, self.precision, self.recall))


        best_val_acc = np.max(self.history.history['val_acc'])
        if self.precision >= 0.75 and self.recall >= 0.85:
            self.save_history_and_conf_mat()
            flag = False

    def exec_v2(self,normal_emotions,normal_not_emotions,mild_emotions=None): # 已经加载好数据，不需要再加载
        global flag
        split_number = -1
        self.loadExternalFiles(normal_emotions,normal_not_emotions,split_number)  # 从保存好的npy文件中读取训练集和测试集
        # self.loadExternalFiles_for_three(normal_emotions,normal_not_emotions,mild_emotions,split_number)  # 从保存好的npy文件中读取训练集和测试集
        print(self.train_set.shape)
        print('transforming data')
        self.augumentDataSets()  # 数据预处理：补齐（多裁少补）
        print(self.train_set.shape)
        self.toNpArrs()
        print(self.train_set.shape)
        print(self.train_labels.shape)
        save_dir = "./all_data/"
        n_classes = len(self.k2l.keys())
        print("n_classes is {}".format(n_classes))
        self.save_files(save_dir, str(n_classes) + '-' + self.type)

        # n_classes = 50
        # self.load_features(n_classes)
        # RocAuc = RocAucEvaluation(validation_data=(self.train_set, self.train_labels), interval=1)

        self.model = self.buildRNN(n_classes)
        print('Starting training')
        filepath = 'E:/ocr_data/model/' + str(n_classes) + "-" + self.type + "-RNNmodel.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.history = self.model.fit(self.train_set, self.train_labels,
                                      validation_data=(self.test_set, self.test_labels),
                                      batch_size=batch_size, epochs=epochs, verbose=1,
                                      shuffle=True, callbacks=callbacks_list)

        # self.history = self.model.fit(self.train_set, self.train_labels,
        #                               validation_data=(self.test_set, self.test_labels),
        #                               batch_size=batch_size, epochs=epochs, verbose=1,
        #                               shuffle=True, callbacks=RocAuc)
        # self.model.save("RNNmodel.h5")

        # self.plotHistory()
        self.conf_mat, self.f1, self.precision, self.recall = self.plot_confuse(n_classes)
        print("self.f1,self.precision,self.recall is {},{},{}".format(self.f1, self.precision, self.recall))

        best_val_acc = np.max(self.history.history['val_acc'])
        if self.precision >= 0.75 and self.recall >= 0.85:
            self.save_history_and_conf_mat()
            flag = False

    def save_history_and_conf_mat(self):
        with open('E:/ocr_data/model/' + self.type + '-' + str(int(time())) + '-history.npy', 'wb') as file_txt:
            pickle.dump(self.history.history, file_txt)
        with open('E:/ocr_data/model/' + self.type + '-' + str(int(time())) + '-confuse.npy', 'wb') as file_txt:
            # pickle.dump(self.conf_mat, file_txt)
            data = (self.conf_mat,self.f1,self.precision,self.recall)
            pickle.dump(data, file_txt)


    def exec_fit_generator(self):


        # Datasets
        train_indexs = np.load("./all_data/train_indexs.npy", allow_pickle=True)
        train_labels = np.load("./all_data/train_labels.npy", allow_pickle=True)
        train_files = np.load("./all_data/train_files.npy", allow_pickle=True)
        train_position = np.load("./all_data/train_position.npy", allow_pickle=True)

        test_indexs = np.load("./all_data/test_indexs.npy", allow_pickle=True)
        test_labels = np.load("./all_data/test_labels.npy", allow_pickle=True)
        test_files = np.load("./all_data/test_files.npy", allow_pickle=True)
        test_position = np.load("./all_data/test_position.npy", allow_pickle=True)

        n_classes = np.max(test_labels) + 1
        print(n_classes)

        # Parameters
        params = {'dim': (data_fixed_length, 6),
                  'batch_size': batch_size,
                  'n_classes': n_classes,
                  'shuffle': True}

        # Generators
        training_generator = DataGenerator(train_indexs, train_labels, train_files, train_position, data_fixed_length,**params)
        validation_generator = DataGenerator(test_indexs, test_labels, test_files, test_position, data_fixed_length,**params)

        self.model = self.buildRNN_for_fit_generator(n_classes)
        self.model.summary()
        print('Starting training')
        self.history = self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=12,epochs=epochs, verbose=1)
        # self.history = self.model.fit_generator(generator=training_generator,
        #                                         validation_data=validation_generator,
        #                                         use_multiprocessing=False,
        #                                         epochs=epochs, verbose=1)
        self.model.save("fit_generator_RNNmodel.h5")


    def buildRNN(self,number_of_classes):
        print('Building model')
        model = Sequential()
        model.add(Bidirectional(GRU(100, return_sequences=True), merge_mode='sum'))
        # model.add(Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.8, return_sequences=True,kernel_regularizer=regularizers.l2(0.001)), merge_mode='sum'))
        model.add(Dropout(0.9))
        model.add(Bidirectional(GRU(100, return_sequences=True), merge_mode='sum'))
        # model.add(Bidirectional(GRU(100,dropout=0.5, recurrent_dropout=0.8, return_sequences=True,kernel_regularizer=regularizers.l2(0.001)), merge_mode='sum'))
        model.add(Dropout(0.8))
        # model.add(SeqSelfAttention(attention_activation='softmax'))
        model.add(Flatten())
        model.add(Dense(number_of_classes, activation='softmax'))
        # model.add(Dense(number_of_classes, activation='sigmoid'))
        print('compiling model')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def buildRNN_for_fit_generator(self,n_classes):
        print('Building model')
        model = Sequential()
        model.add(Bidirectional(GRU(500, return_sequences=True), merge_mode='sum',input_shape=(data_fixed_length, n_dimensions)))
        model.add(Bidirectional(GRU(300, return_sequences=True), merge_mode='sum'))
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax'))
        print('compiling model')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def toNpArrs(self):
        def toNpArr(s):
            new = []
            for n in tqdm(range(len(s))):
                i = s[n]
                temp = []
                for j in i:
                    temp.append(np.asarray(j))
                new.append(np.asarray(temp))
            return np.array(new)

        self.train_set = toNpArr(self.train_set)
        self.test_set = toNpArr(self.test_set)

    def testToNpArrs(self):
        def toNpArr(s):
            new = []
            for n in tqdm(range(len(s))):
                i = s[n]
                temp = []
                for j in i:
                    temp.append(np.asarray(j))
                new.append(np.asarray(temp))
            return np.array(new)

        # self.train_set = toNpArr(self.train_set)
        self.test_set = toNpArr(self.test_set)

    def loadTestset(self,testX):
        self.test_set = testX

    def augumentDataSets(self):
        def augumentDataSet(dataset):
            for i in tqdm(range(len(dataset))):
                if len(dataset[i]) > data_fixed_length:
                    dataset[i] = dataset[i][:data_fixed_length]
                else:
                    dataset[i] = dataset[i] + [[0] * 6] * (data_fixed_length - len(dataset[i]))

        augumentDataSet(self.train_set)
        augumentDataSet(self.test_set)

    def buildInternalRepresentationsFromDic(self, train_dic, test_dic):
        '''this build representation file from files built by IO.py'''
        print("building dictionary from  IO class")

        def buildInternalRepresentation(dic):
            labels = []
            samples = [[0]]
            n_keys = len(dic.keys())
            print("number of keys in dic:", n_keys)

            for key in dic.keys():
                for sample in dic[key]:
                    labels.append(key)
                    new_strokes = [[]]
                    for stroke in sample:
                        # [x,y,dx,dy,pen down, pen up]
                        stroke_rep = []
                        if len(stroke) == 1:
                            stroke_rep = [[stroke[0][0], stroke[0][1], 0, 0, 1, 1]]
                        else:
                            for i in range(0, len(stroke) - 1):
                                c = stroke[i]
                                n = stroke[i + 1]
                                stroke_rep.append([int(c[0]), int(c[1]), int(n[0] - c[0]), int(n[1] - c[1]), 0, 0])
                            # pen down stroke
                            stroke_rep[0] = [int(stroke_rep[0][0]), int(stroke_rep[0][1]), int(stroke_rep[0][2]),
                                             int(stroke_rep[0][3]), 1, stroke_rep[0][5]]
                            # pen up stroke
                            stroke_rep[-1] = [int(stroke_rep[-1][0]), int(stroke_rep[-1][1]), int(stroke_rep[-1][2]),
                                              int(stroke_rep[-1][3]), stroke_rep[-1][4], 1]
                        new_strokes += stroke_rep
                    samples += [new_strokes[1:]]
            return samples[1:], labels

        self.train_set, self.train_labels = buildInternalRepresentation(train_dic)
        self.test_set, self.test_labels = buildInternalRepresentation(test_dic)
        print(
            "internal representation has been built, call self.saveInternalRepresentationFiles() to save these representation")
        print("self.loadInternalRepresentationFiles() will load the saved files to RNN ")

    def buildInternalRepresentationsFromDicV2(self, chars_dic):
        '''this build representation file from files built by IO.py'''
        print("building dictionary from  IO class")

        def buildInternalRepresentation(dic):
            labels = []
            samples = [[0]]
            n_keys = len(dic.keys())
            print("number of keys in dic:", n_keys)

            for key in dic.keys():
                for sample in dic[key]:
                    labels.append(key)
                    new_strokes = [[]]
                    for stroke in sample:
                        # [x,y,dx,dy,pen down, pen up]
                        stroke_rep = []
                        if len(stroke) == 1:
                            stroke_rep = [[stroke[0][0], stroke[0][1], 0, 0, 1, 1]]
                        else:
                            for i in range(0, len(stroke) - 1):
                                c = stroke[i]
                                n = stroke[i + 1]
                                stroke_rep.append([int(c[0]), int(c[1]), int(n[0] - c[0]), int(n[1] - c[1]), 0, 0])
                            # pen down stroke
                            stroke_rep[0] = [int(stroke_rep[0][0]), int(stroke_rep[0][1]), int(stroke_rep[0][2]),
                                             int(stroke_rep[0][3]), 1, stroke_rep[0][5]]
                            # pen up stroke
                            stroke_rep[-1] = [int(stroke_rep[-1][0]), int(stroke_rep[-1][1]), int(stroke_rep[-1][2]),
                                              int(stroke_rep[-1][3]), stroke_rep[-1][4], 1]
                        new_strokes += stroke_rep
                    samples += [new_strokes[1:]]
            return samples[1:], labels

        data_set, labels = buildInternalRepresentation(chars_dic)
        return data_set, labels

    def saveInternalRepresentationFiles(self):
        start = time()
        print('saving representation files...')
        np.save('trainset', self.train_set)
        np.save('trainlabels', self.train_labels)
        np.save('testset', self.test_set)
        np.save('testlabel', self.test_labels)
        print("4 representation files saved in", time() - start, "seconds")

    def loadInternalRepresentationFiles(self):
        start = time()
        sava_dir = "./all_data"
        print('reading representation files...')
        # self.train_set = np.load("trainset.npy", allow_pickle=True)
        # self.train_labels = np.load('trainlabels.npy', allow_pickle=True)
        # self.test_set = np.load('testset.npy', allow_pickle=True)
        # self.test_labels = np.load('testlabel.npy', allow_pickle=True)
        # print("4 np files read in", time() - start, "seconds")
        # print(self.train_set.shape)
        # print(self.test_set.shape)

        type = self.type
        sample_step = 300
        normal_path = 'E:/mer-database/av/normal/'
        normal_not_path = 'E:/mer-database/av/normal_not/'
        normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
        normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
        augment_factor_normal = 10
        normal_emotions,normal_peoples_id,normal_file_number = get_data_from_csv(normal_path, type,augment_factor_normal,sample_step)
        augment_factor_normal_not = 10
        normal_not_emotions,not_peoples_id,not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not,sample_step)
        print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
        normal_emotions = [e[int(len(e)/2):] for e in normal_emotions]  # 取后半段
        normal_not_emotions = [e[int(len(e)/2):] for e in normal_not_emotions] # 取后半段
        normal_samples, normal_labels = parse_data(normal_emotions, 0)
        # indices1 = shuffle_indices(augment_factor_normal, len(normal_samples))
        normal_not_samples, normal_not_labels = parse_data(normal_not_emotions, 1)
        # indices2 = shuffle_indices(augment_factor_normal_not,len(normal_not_samples))
        all_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples),), dtype=object)
        label_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples),), dtype=object)
        for i in tqdm(range(len(normal_samples))):
            all_mats[i] = normal_samples[i]
            label_mats[i] = str(normal_labels[i])
        for n in tqdm(range(len(normal_not_samples))):
            all_mats[i+n+1] = normal_not_samples[n]
            label_mats[i+n+1] = str(normal_not_labels[n])

        # indices = np.arange(all_mats.shape[0]*augment_factor)
        # np.random.shuffle(indices)
        # indices = shuffle_indices(augment_factor, all_mats.shape[0])
        # indices2 = [i + len(indices1) for i in indices2]
        # indices = indices1 + indices2
        indices = shuffle_indices_v2(augment_factor_normal,len(normal_emotions),augment_factor_normal_not,len(normal_not_emotions))
        rate = 0.9
        split_index = int(rate * all_mats.shape[0])
        train_set = all_mats[indices[:split_index]]
        test_set = all_mats[indices[split_index:]]
        train_label = label_mats[indices[:split_index]]
        test_label = label_mats[indices[split_index:]]

        self.train_set, self.train_labels, self.test_set, self.test_labels = train_set,train_label,test_set,test_label


        self.convertLabelsToKeys() # label to key; key to label,由于key是标签的gbk2312字符串，为了便于训练，将其转换成数字
        self.train_labels = np.reshape(np.array(self.train_labels), (len(self.train_labels), 1))
        self.test_labels = np.reshape(np.array(self.test_labels), (len(self.test_labels), 1))
        print("representation files have been loaded")
        print("call self.exec() to start training.")
        print("model configuration can be modified in buildRNN()")

    def loadExternalFiles(self,normal_emotions,normal_not_emotions,split_number=-1):
        start = time()
        sava_dir = "./all_data"
        print('reading representation files...')
        # self.train_set = np.load("trainset.npy", allow_pickle=True)
        # self.train_labels = np.load('trainlabels.npy', allow_pickle=True)
        # self.test_set = np.load('testset.npy', allow_pickle=True)
        # self.test_labels = np.load('testlabel.npy', allow_pickle=True)
        # print("4 np files read in", time() - start, "seconds")
        # print(self.train_set.shape)
        # print(self.test_set.shape)

        type = self.type
        sample_step = 300
        normal_path = 'E:/mer-database/av/normal/'
        normal_not_path = 'E:/mer-database/av/normal_not/'
        normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
        normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
        augment_factor_normal = 10
        # normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type,augment_factor_normal, sample_step)
        augment_factor_normal_not = 10
        augment_factor_mild = 10
        # normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
        # print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
        if split_number == -1:
            normal_emotions = [e[int(len(e) / 2):] for e in normal_emotions]  # 取后半段
            normal_not_emotions = [e[int(len(e) / 2):] for e in normal_not_emotions]  # 取后半段
        else:
            no_splits = 20  # 窗口个数
            normal_emotions = [e[int(len(e) / no_splits * (split_number + 0.5)):int(len(e) / no_splits * (split_number+1.5))] for e in normal_emotions]  # 取后半段
            normal_not_emotions = [e[int(len(e) / no_splits * (split_number + 0.5)):int(len(e) / no_splits * (split_number+1.5))] for e in normal_not_emotions]  # 取后半段

        normal_samples, normal_labels = parse_data(normal_emotions, 0)
        # indices1 = shuffle_indices(augment_factor_normal, len(normal_samples))
        normal_not_samples, normal_not_labels = parse_data(normal_not_emotions, 1)
        # indices2 = shuffle_indices(augment_factor_normal_not,len(normal_not_samples))
        all_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples),), dtype=object)
        label_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples),), dtype=object)
        for i in tqdm(range(len(normal_samples))):
            all_mats[i] = normal_samples[i]
            label_mats[i] = str(normal_labels[i])
        for n in tqdm(range(len(normal_not_samples))):
            all_mats[i + n + 1] = normal_not_samples[n]
            label_mats[i + n + 1] = str(normal_not_labels[n])


        # indices = np.arange(all_mats.shape[0]*augment_factor)
        # np.random.shuffle(indices)
        # indices = shuffle_indices(augment_factor, all_mats.shape[0])
        # indices2 = [i + len(indices1) for i in indices2]
        # indices = indices1 + indices2
        indices = shuffle_indices_v2(augment_factor_normal, len(normal_emotions), augment_factor_normal_not, len(normal_not_emotions))
        # indices = shuffle_indices_v3(augment_factor_normal, len(normal_emotions), augment_factor_normal_not, len(normal_not_emotions),augment_factor_mild, len(mild_emotions))
        rate = 0.9
        split_index = int(rate * all_mats.shape[0])
        train_set = all_mats[indices[:split_index]]
        test_set = all_mats[indices[split_index:]]
        train_label = label_mats[indices[:split_index]]
        test_label = label_mats[indices[split_index:]]

        self.train_set, self.train_labels, self.test_set, self.test_labels = train_set, train_label, test_set, test_label

        self.convertLabelsToKeys()  # label to key; key to label,由于key是标签的gbk2312字符串，为了便于训练，将其转换成数字
        self.train_labels = np.reshape(np.array(self.train_labels), (len(self.train_labels), 1))
        self.test_labels = np.reshape(np.array(self.test_labels), (len(self.test_labels), 1))
        print("representation files have been loaded")
        print("call self.exec() to start training.")
        print("model configuration can be modified in buildRNN()")

    def loadExternalFiles_for_three(self,normal_emotions,normal_not_emotions,mild_emotions,split_number=-1):
        start = time()
        sava_dir = "./all_data"
        print('reading representation files...')
        # self.train_set = np.load("trainset.npy", allow_pickle=True)
        # self.train_labels = np.load('trainlabels.npy', allow_pickle=True)
        # self.test_set = np.load('testset.npy', allow_pickle=True)
        # self.test_labels = np.load('testlabel.npy', allow_pickle=True)
        # print("4 np files read in", time() - start, "seconds")
        # print(self.train_set.shape)
        # print(self.test_set.shape)

        type = self.type
        sample_step = 300
        normal_path = 'E:/mer-database/av/normal/'
        normal_not_path = 'E:/mer-database/av/normal_not/'
        normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
        normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
        augment_factor_normal = 10
        # normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type,augment_factor_normal, sample_step)
        augment_factor_normal_not = 10
        augment_factor_mild = 10
        # normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
        # print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
        if split_number == -1:
            normal_emotions = [e[int(len(e) / 2):] for e in normal_emotions]  # 取后半段
            normal_not_emotions = [e[int(len(e) / 2):] for e in normal_not_emotions]  # 取后半段
            mild_emotions = [e[int(len(e) / 2):] for e in mild_emotions]  # 取后半段
        else:
            no_splits = 20  # 窗口个数
            normal_emotions = [e[int(len(e) / no_splits * (split_number + 0.5)):int(len(e) / no_splits * (split_number+1.5))] for e in normal_emotions]  # 取后半段
            normal_not_emotions = [e[int(len(e) / no_splits * (split_number + 0.5)):int(len(e) / no_splits * (split_number+1.5))] for e in normal_not_emotions]  # 取后半段
            mild_emotions = [e[int(len(e) / no_splits * (split_number + 0.5)):int(len(e) / no_splits * (split_number+1.5))] for e in mild_emotions]  # 取后半段
        normal_samples, normal_labels = parse_data(normal_emotions, 0)
        # indices1 = shuffle_indices(augment_factor_normal, len(normal_samples))
        normal_not_samples, normal_not_labels = parse_data(normal_not_emotions, 1)
        mild_samples, mild_labels = parse_data(mild_emotions, 2)
        # indices2 = shuffle_indices(augment_factor_normal_not,len(normal_not_samples))
        all_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples) + len(mild_samples),), dtype=object)
        label_mats = np.ndarray(shape=(len(normal_samples) + len(normal_not_samples) + len(mild_samples),), dtype=object)
        for i in tqdm(range(len(normal_samples))):
            all_mats[i] = normal_samples[i]
            label_mats[i] = str(normal_labels[i])
        for n in tqdm(range(len(normal_not_samples))):
            all_mats[i + n + 1] = normal_not_samples[n]
            label_mats[i + n + 1] = str(normal_not_labels[n])
        for j in tqdm(range(len(mild_samples))):
            all_mats[i + n + j + 2] = mild_samples[j]
            label_mats[i + n + j + 2] = str(mild_labels[j])

        # indices = np.arange(all_mats.shape[0]*augment_factor)
        # np.random.shuffle(indices)
        # indices = shuffle_indices(augment_factor, all_mats.shape[0])
        # indices2 = [i + len(indices1) for i in indices2]
        # indices = indices1 + indices2
        # indices = shuffle_indices_v2(augment_factor_normal, len(normal_emotions), augment_factor_normal_not, len(normal_not_emotions))
        indices = shuffle_indices_v3(augment_factor_normal, len(normal_emotions), augment_factor_normal_not, len(normal_not_emotions),augment_factor_mild, len(mild_emotions))
        rate = 0.9
        split_index = int(rate * all_mats.shape[0])
        train_set = all_mats[indices[:split_index]]
        test_set = all_mats[indices[split_index:]]
        train_label = label_mats[indices[:split_index]]
        test_label = label_mats[indices[split_index:]]

        self.train_set, self.train_labels, self.test_set, self.test_labels = train_set, train_label, test_set, test_label

        self.convertLabelsToKeys()  # label to key; key to label,由于key是标签的gbk2312字符串，为了便于训练，将其转换成数字
        self.train_labels = np.reshape(np.array(self.train_labels), (len(self.train_labels), 1))
        self.test_labels = np.reshape(np.array(self.test_labels), (len(self.test_labels), 1))
        print("representation files have been loaded")
        print("call self.exec() to start training.")
        print("model configuration can be modified in buildRNN()")

    def convertLabelsToKeys(self): #
        def defineDict(labels):
            label_dic = {}
            current_index = 0
            for l in labels:
                if l not in label_dic.keys():
                    label_dic[l] = current_index
                    current_index += 1
            return label_dic, {v: k for k, v in label_dic.items()}

        # self.l2k, self.k2l = defineDict(self.test_labels)
        all_labels = np.hstack((self.train_labels.copy(), self.test_labels.copy()))
        self.l2k, self.k2l = defineDict(all_labels)

        def toClassArray(labels):
            new_labels = [[]]
            for l in labels:
                new_i = [0] * number_of_classes
                new_i[l] = 1
                new_labels.append(new_i)
            return new_labels[1:]

        self.train_labels = [self.l2k[i] for i in self.train_labels]
        self.test_labels = [self.l2k[i] for i in self.test_labels]

    def show(self, i):
        for stroke in self.train_set[i]:
            if stroke[-2] == 1:
                new_stroke = []
            if stroke[-1] == 1:
                new_stroke.append([stroke[0], stroke[1]])
                plt.plot([x[0] for x in new_stroke], [x[1] for x in new_stroke])
            else:
                new_stroke.append([stroke[0], stroke[1]])
        plt.show()



    def plotHistory(self):
        # list all data in history
        history = self.history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_confuse(self,n_classes):
        # n_classes = 2
        filepath = 'E:/ocr_data/model/' + str(n_classes) + "-" + self.type + "-RNNmodel.hdf5"
        model = load_model(filepath)
        # model = self.model
        # model.summary()
        conf_mat,f1,precision,recall = plot_confuse(model, self.test_set, self.test_labels)
        return conf_mat,f1,precision,recall

    def predictTestData(self, modelPath, filePath, type):

        model = load_model(modelPath)
        augment_factor_normal = 1
        sample_step = 300
        test_emotions, test_peoples_id, test_file_number = get_data_from_csv(filePath, type, augment_factor_normal,sample_step)
        test_emotions = [e[int(len(e) / 2):] for e in test_emotions]  # 取后半段
        test_samples, test_labels = parse_data(test_emotions, 0)
        all_mats = np.ndarray(shape=(len(test_samples),), dtype=object)
        label_mats = np.ndarray(shape=(len(test_samples),), dtype=object)
        for i in tqdm(range(len(test_samples))):
            all_mats[i] = test_samples[i]
            label_mats[i] = str(test_labels[i])

        test_set = all_mats
        test_label = label_mats

        self.test_set, self.test_labels = test_set, test_label

        self.convertLabelsToKeys()  # label to key; key to label,由于key是标签的gbk2312字符串，为了便于训练，将其转换成数字
        self.test_labels = np.reshape(np.array(self.test_labels), (len(self.test_labels), 1))

        self.augumentDataSets()  # 数据预处理：补齐（多裁少补）
        print(self.test_set.shape)
        self.toNpArrs()
        print(self.test_set.shape)
        print(self.test_labels.shape)

        y_pred_test = model.predict(self.test_set)
        max_y_pred_test = np.argmax(y_pred_test, axis=1)
        print(max_y_pred_test)

    def save_files(self,save_dir,n_classes):
        start = time()
        np.save(save_dir + str(n_classes) + '-all_train_set.npy', self.train_set)
        np.save(save_dir + str(n_classes) + '-all_train_labels.npy', self.train_labels)
        np.save(save_dir + str(n_classes) + '-all_test_set.npy', self.test_set)
        np.save(save_dir + str(n_classes) + '-all_test_labels.npy', self.test_labels)
        np.save(save_dir + str(n_classes) + '-k2l.npy', self.k2l)
        np.save(save_dir + str(n_classes) + '-l2k.npy', self.l2k)
        print("4 representation files saved in", time() - start, "seconds")

    def load_features(self,n_classes):
        start = time()
        sava_dir = "./all_data/"
        print('reading representation files...')
        # self.train_set = np.load("trainset.npy", allow_pickle=True)
        # self.train_labels = np.load('trainlabels.npy', allow_pickle=True)
        # self.test_set = np.load('testset.npy', allow_pickle=True)
        # self.test_labels = np.load('testlabel.npy', allow_pickle=True)
        self.train_set = np.load(sava_dir + str(n_classes) + '-all_train_set.npy', allow_pickle=True)
        self.train_labels = np.load(sava_dir + str(n_classes) + '-all_train_labels.npy', allow_pickle=True)
        self.test_set = np.load(sava_dir + str(n_classes) + '-all_test_set.npy', allow_pickle=True)
        self.test_labels = np.load(sava_dir + str(n_classes) + '-all_test_labels.npy', allow_pickle=True)
        print("4 np files read in", time() - start, "seconds")
        print(self.train_set.shape)
        print(type(self.train_set))
        print(self.train_set.dtype)
        # t = self.train_set[0]
        # print(type(t))
        # print(t)
        # a = self.train_labels[1100]
        # a = a.replace('0x', '')
        # a_bytes = bytes.fromhex(a)
        # print(a_bytes)
        # utf8_decode = a_bytes.decode("gbk")
        # print(utf8_decode)
        # print(self.train_labels[10].decode('UTF-8'))
        # unicode = unicode(self.train_labels[10],'gb2312')
        # self.show(1100)
        # self.convertLabelsToKeys() # label to key; key to label,由于key是标签的gbk2312字符串，为了便于训练，将其转换成数字
        # self.train_labels = np.reshape(np.array(self.train_labels), (len(self.train_labels), 1))
        # self.test_labels = np.reshape(np.array(self.test_labels), (len(self.test_labels), 1))
        print("representation files have been loaded")
        print("call self.exec() to start training.")
        print("model configuration can be modified in buildRNN()")


def continueTraining(batch_size, n_epoch):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=0,
        verbose=0,
        mode='auto'
    )
    rnn = RNN()
    rnn.loadInternalRepresentationFiles()
    print('transforming data')
    rnn.augumentDataSets()
    rnn.toNpArrs()
    print('continue training')
    model = load_model("RNNmodel.h5")
    rnn.history = model.fit(rnn.train_set, rnn.train_labels, validation_data=(rnn.test_set, rnn.test_labels),
                            batch_size=batch_size, epochs=n_epoch, verbose=1, callbacks=[early_stopping()])
    model.save("RNNmodel.h5")

def get_root_path():
    current_file_path = os.getcwd()
    project_root_path = None
    # print("current_file_path:\t" + current_file_path)
    index = 0
    for path in sys.path:
        # print("sys_path%s:\t\t\t" % index + path)
        index += 1
        if current_file_path == path:
            continue

        if current_file_path.__contains__(path):
            project_root_path = path
            break

    if project_root_path is None:
        # 如果未获取到，说明当前路径为根路径
        project_root_path = current_file_path

        # 替换斜杠
        project_root_path = project_root_path.replace("\\", "/")

    return project_root_path

def predictFiles(test_set, test_labels):
    # test_set = np.load('testset.npy', allow_pickle=True)
    # test_labels = np.load('testlabel.npy', allow_pickle=True)
    path = get_root_path()
    # print(path)
    model = load_model(path + "/RNNmodel.h5")
    y_pred_test = model.predict(test_set)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    print(max_y_pred_test.shape)
    print(max_y_pred_test)
    return max_y_pred_test
    # # Take the class with the highest probability from the test predictions
    # max_y_pred_test = np.argmax(y_pred_test, axis=1)
    # rnn = RNN()
    # rnn.loadTestset(test_set)
    # rnn.toNpArrs()
    # model = load_model("RNNmodel.h5")
    # y_pred_test = model.predict(rnn.test_set)

def predictFilesWithModel(test_set,model,N=3):
    # test_set = np.load('testset.npy', allow_pickle=True)
    # test_labels = np.load('testlabel.npy', allow_pickle=True)
    path = get_root_path()
    # print(path)
    # model = load_model(path + "/RNNmodel.h5")
    y_pred_test = model.predict(test_set)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    # print(max_y_pred_test.shape)
    # print(max_y_pred_test)
    # return max_y_pred_test
    import heapq
    # N = 3
    max_N_idxs = []
    for y in y_pred_test:
        max_N_idx = heapq.nlargest(N, range(len(y)), y.__getitem__)
        max_N_idxs.append(max_N_idx)
    # pred_N = [tmp[i] for i in max_N_idx]
    # print("pred N is: {}".format(pred_N))
    return max_y_pred_test, max_N_idxs
    # # Take the class with the highest probability from the test predictions
    # max_y_pred_test = np.argmax(y_pred_test, axis=1)
    # rnn = RNN()
    # rnn.loadTestset(test_set)
    # rnn.toNpArrs()
    # model = load_model("RNNmodel.h5")
    # y_pred_test = model.predict(rnn.test_set)


def get_data_from_csv(path,type,augment_factor,sample_step=200):
    abled_types = ['01', '02', '03', '04', '05', '06']
    type_names = ['悲伤', '厌恶', '愤怒', '恐惧', '中性', '开心']
    emotions = []
    peoples_id = []
    smaple_flag = True
    # smaple_flag = False
    # sample_step = 200
    filterd = True
    medfilt_flag = True
    file_number = 0
    all_video_infos = pd.read_csv('F:/PycharmProjects/emonet-master/output/2022/video_info/video_info.csv', encoding='ANSI')
    all_people_name_sheet = all_video_infos['people_name_sheet'].values.tolist()
    all_fps = all_video_infos['fps'].values.tolist()
    all_frame_counter = all_video_infos['frame_counter'].values.tolist()
    all_duration = all_video_infos['duration'].values.tolist()
    for home, dirs, files in os.walk(path):
        # 获取xlsx文件信息
        if len(files) > 0:
            filenames = [filename for filename in files if filename.find('csv') > 0]

        for fs in filenames:
            details = fs.split("-")
            people_id = "-".join(fs.split("-")[:-2])
            # person_id = details[0] + "-" + details[1]
            file_type = details[-2][-2:]
            if type == file_type or type_names[int(type)-1] == file_type:
                filename = os.path.join(home,fs)
                print(filename)
                data = pd.read_csv(filename, encoding='ANSI')
                arousal = list(data['disp_arousal'].values)
                valence = list(data['disp_valence'].values)
                intensity = list(data['disp_intensity'].values)

                if filterd:
                    zero_positions = [i for i in range(len(arousal)) if arousal[i] == 0.0 and valence[i] == 0.0 and intensity[i] == 0]
                    arousal = [arousal[i] for i in range(len(arousal)) if i not in zero_positions]
                    valence = [valence[i] for i in range(len(valence)) if i not in zero_positions]
                    intensity = [intensity[i] for i in range(len(intensity)) if i not in zero_positions]
                # arousal = [arousal[i] if arousal[i] != 0 else arousal[i-1] for i in range(1,len(arousal)-1)]
                # valence = [valence[i] if valence[i] != 0 else valence[i - 1] for i in range(1, len(valence) - 1)]
                # intensity = [intensity[i] if intensity[i] != 0 else intensity[i - 1] for i in range(1, len(intensity) - 1)]
                if medfilt_flag:
                    arousal = signal.medfilt(arousal, 125)
                    valence = signal.medfilt(valence, 125)
                    intensity = signal.medfilt(intensity, 125)
                    # arousal = signal.savgol_filter(arousal, 53,3)

                #获取fps等信息
                pns = people_id + "-" + file_type
                pns_data = [i for i,p in enumerate(all_people_name_sheet) if p.find(pns) != -1]
                if len(pns_data) > 0:
                    # video_index = all_people_name_sheet[pns_data[0]]
                    # fps = all_fps[video_index]
                    fps = all_fps[pns_data[0]]
                else:
                    fps = 30
                    print("fps no exist: " +  pns)
                lower, upper = 5, int(fps*3)
                # augment_factor = 10

                if sample_step > len(arousal) - lower:
                    print("continue")
                    continue

                # if smaple_flag:
                #随机生成n个(lower, upper)范围内不重复的数字
                # rand_samples = random.sample(range(lower, upper), augment_factor)
                rand_samples = range(lower, upper, int((upper - lower)/augment_factor))
                rand_samples = [random.randint(-int(0.25*(upper - lower)/augment_factor),int(0.25*(upper - lower)/augment_factor)) + rand_samples[n] for n in range(augment_factor)]
                for n in range(augment_factor):
                    # random_integer = random.randint(lower, upper)
                    random_integer = rand_samples[n]
                    arousal_sample_positions = np.linspace(random_integer, len(arousal)-1, sample_step, endpoint=False)
                    arousal_sample_positions = [int(s) for s in arousal_sample_positions]
                    # print("a,s,m is {},{},{}".format(len(arousal),np.max(arousal_sample_positions),arousal_sample_positions))
                    # arousal_tmp = [arousal[i] for i in arousal_sample_positions]
                    arousal_tmp = [arousal[i]*1000 for i in arousal_sample_positions]
                    valence_sample_positions = np.linspace(random_integer, len(valence)-1, sample_step, endpoint=False)
                    valence_sample_positions = [int(s) for s in valence_sample_positions]
                    # print("v,s,m is {},{},{}".format(len(valence), np.max(valence_sample_positions),valence_sample_positions))
                    # valence_tmp = [valence[i] for i in valence_sample_positions]
                    valence_tmp = [valence[i]*1000 for i in valence_sample_positions]
                    # intensity_sample_positions = np.linspace(random_integer, len(intensity)-1, sample_step, endpoint=False)
                    # intensity_sample_positions = [int(s) for s in intensity_sample_positions]
                    # intensity = [intensity[i] for i in intensity_sample_positions]
                    # tmp = [arousal,valence,intensity]
                    tmp = [(arousal_tmp[i],valence_tmp[i]) for i in range(len(arousal_tmp))]
                    emotions.append(tmp)
                    peoples_id.append(people_id)
                file_number += 1
            # if len(emotions) > 15:
            #     break
    # all_mats = np.ndarray(shape=(len(emotions),), dtype=object)
    # for i in range(len(emotions)):
    #     all_mats[i] = emotions[i]
    # emotions = np.array(emotions)
    return emotions,peoples_id,file_number

# [x,y,dx,dy,pen down, pen up]
def parse_data(emotions,key):
    labels = []
    samples = [[0]]
    new_strokes = [[]]
    for em in emotions:
        stroke_rep = []
        for i in range(0, len(em) - 1):
            c = em[i]
            n = em[i + 1]
            stroke_rep.append([c[0], c[1], n[0] - c[0], n[1] - c[1], 0, 0])
        samples += [stroke_rep]
        labels.append(key)
    return samples[1:], labels

def shuffle_indices(augment_factor,total_len):
    #augment_factor = 20
    tmp = range(0, total_len + 1, augment_factor)
    tmp = list(tmp)
    #print(tmp)
    a = [list(range(tmp[i], tmp[i + 1])) for i in range(len(tmp) - 1)]
    #print(a)
    #print(len(a))
    indices = np.arange(len(a))
    #print(indices)
    np.random.shuffle(indices)
    #print(indices)
    b = []
    for i in indices:
        b += a[i]
    #print(b)
    return b

def shuffle_indices_v2(augment_factor1,total_len1,augment_factor2,total_len2):
    #augment_factor = 20
    tmp1 = range(0, total_len1 + 1, augment_factor1)
    tmp1 = list(tmp1)
    #print(tmp)
    a = [list(range(tmp1[i], tmp1[i + 1])) for i in range(len(tmp1) - 1)]

    tmp2 = range(0, total_len2 + 1, augment_factor2)
    tmp2 = list(tmp2)
    # print(tmp)
    b = [list(range(tmp2[i], tmp2[i + 1])) for i in range(len(tmp2) - 1)]
    b = [[i + total_len1 for i in t] for t in b]
    a = a + b
    #print(a)
    #print(len(a))
    indices = np.arange(len(a))
    #print(indices)
    np.random.shuffle(indices)
    #print(indices)
    b = []
    for i in indices:
        b += a[i]
    #print(b)
    return b

def shuffle_indices_v3(augment_factor1,total_len1,augment_factor2,total_len2,augment_factor3,total_len3):
    #augment_factor = 20
    tmp1 = range(0, total_len1 + 1, augment_factor1)
    tmp1 = list(tmp1)
    #print(tmp)
    a = [list(range(tmp1[i], tmp1[i + 1])) for i in range(len(tmp1) - 1)]

    tmp2 = range(0, total_len2 + 1, augment_factor2)
    tmp2 = list(tmp2)
    # print(tmp)
    b = [list(range(tmp2[i], tmp2[i + 1])) for i in range(len(tmp2) - 1)]
    b = [[i + total_len1 for i in t] for t in b]

    tmp3 = range(0, total_len3 + 1, augment_factor3)
    tmp3 = list(tmp3)
    # print(tmp)
    c = [list(range(tmp3[i], tmp3[i + 1])) for i in range(len(tmp3) - 1)]
    c = [[i + total_len1 + total_len2 for i in t] for t in c]

    a = a + b + c
    #print(a)
    #print(len(a))
    indices = np.arange(len(a))
    #print(indices)
    np.random.shuffle(indices)
    #print(indices)
    b = []
    for i in indices:
        b += a[i]
    #print(b)
    return b

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 https://blog.csdn.net/weixin_41767802/article/details/108047350

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    # plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score,precision_score,recall_score
    batch = 512
    labels = ['Normal','Dementia']
    predictions = model.predict_classes(x_val, batch_size=batch)
    # truelabel = y_val.argmax(axis=1)  # 将one-hot转化为label
    truelabel = [int(i[0]) for i in y_val]
    truelabel = np.array(truelabel)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    # plt.figure()
    f1 = f1_score(truelabel, predictions)
    precision = precision_score(truelabel, predictions)
    recall = recall_score(truelabel, predictions)
    # f1 = f1_score(truelabel, predictions, average='weighted')
    # precision = precision_score(truelabel, predictions, average='weighted')
    # recall = recall_score(truelabel, predictions, average='weighted')
    print("f1,precision, recall is {},{},{}".format(f1,precision, recall))
    # plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')
    return conf_mat,f1,precision,recall

def read_and_plot_history_conf_mat(history_txt,conf_mat_txt):
    history = None
    with open(history_txt, 'rb') as file_txt:
        history = pickle.load(file_txt)
    # list all data in history
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    conf_mat = None
    labels = ['Normal', 'Dementia']
    with open(conf_mat_txt, 'rb') as file_txt:
        data = pickle.load(file_txt)
        conf_mat = data[0]
        f1,precision, recall = data[1:]
        print("f1,precision, recall is {},{},{}".format(f1,precision, recall))
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')



'''
二分类（正常，痴呆）模型的训练过程
每类视频各自训练，每类视频训练循环50次，模型性能达到预设值则退出循环，预设值可以自定
'''	
def test1():
    flag = True
    video_type = '06'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break

    flag = True
    video_type = '05'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break

    flag = True
    video_type = '04'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break

    flag = True
    video_type = '03'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break

    flag = True
    video_type = '02'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break

    flag = True
    video_type = '01'
    rnn = RNN(video_type)
    i = 0
    while flag:
        rnn.exec()
        i += 1
        if i > 50: break
    exit()

	
'''
三分类（正常，轻度认识功能障碍，痴呆）模型的训练过程
每类视频各自训练，每类视频训练循环50次，模型性能达到预设值则退出循环，预设值可以自定
'''
def test2():
    flag = True
    type = '06'
    sample_step = 300
    normal_path = 'E:/mer-database/av/normal/'
    normal_not_path = 'E:/mer-database/av/normal_not/'
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    mild_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/mild/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    augment_factor_normal = 10
    # mild_emotions, mild_peoples_id, mild_file_number = get_data_from_csv(mild_path, type, augment_factor_normal,sample_step)
    mild_emotions = None
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions,normal_not_emotions,mild_emotions)
        i += 1
        if i > 50: break

    flag = True
    type = '05'
    sample_step = 300
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions, normal_not_emotions)
        i += 1
        if i > 50: break

    flag = True
    type = '04'
    sample_step = 300
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions, normal_not_emotions)
        i += 1
        if i > 50: break

    flag = True
    type = '03'
    sample_step = 300
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions, normal_not_emotions)
        i += 1
        if i > 50: break

    flag = True
    type = '02'
    sample_step = 300
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions, normal_not_emotions)
        i += 1
        if i > 50: break

    flag = True
    type = '01'
    sample_step = 300
    normal_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal/'
    normal_not_path = 'F:/PycharmProjects/emonet-master/output/2022/' + type + '/normal_not/'
    augment_factor_normal = 10
    normal_emotions, normal_peoples_id, normal_file_number = get_data_from_csv(normal_path, type, augment_factor_normal,sample_step)
    augment_factor_normal_not = 10
    normal_not_emotions, not_peoples_id, not_file_number = get_data_from_csv(normal_not_path, type,augment_factor_normal_not, sample_step)
    print("normal_file_number, not_file_number is {},{}".format(normal_file_number, not_file_number))
    rnn = RNN(type)
    i = 0
    while flag:
        rnn.exec_v2(normal_emotions, normal_not_emotions)
        i += 1
        if i > 50: break
    exit()

'''
加载预训练好的模型，对测试样本进行预测
'''
def pred_test():
    video_type = '06'  # 视频类型
    modelPath = 'E:/ocr_data/model/2-06-RNNmodel.hdf5' #模型保存地址
    filePath = 'F:/PycharmProjects/emonet-master/output/2022/' + video_type + '/normal_not/'  #测试样本特征保存地址
    filePath = 'F:/PycharmProjects/emonet-master/output/2022/' + video_type + '/normal/'  #测试样本特征保存地址
    rnn = RNN(video_type) #实例化模型
    rnn.predictTestData(modelPath,filePath,video_type) #进行预测

if __name__ == "__main__":
    # video_type = '06'
    # rnn = RNN(video_type)
    # rnn.exec()
    # exit()
    # flag = True
    # video_type = '06'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    #
    # flag = True
    # video_type = '05'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    #
    # flag = True
    # video_type = '04'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    #
    # flag = True
    # video_type = '03'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    #
    # flag = True
    # video_type = '02'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    #
    # flag = True
    # video_type = '01'
    # rnn = RNN(video_type)
    # rnn.exec()
    # i = 0
    # while flag:
    #     rnn.exec_v2()
    #     i += 1
    #     if i > 50: break
    # exit()

    ###################################################################################
    ###################################################################################
    test1()
    # test2()
    # pred_test()

    exit()
    # # rnn.exec_fit_generator()
    # predictTestData()
    # predictForStrokeTrack()
    history_txt, conf_mat_txt = 'E:/ocr_data/model/05-1645402543-history.npy', 'E:/ocr_data/model/05-1645402543-confuse.npy'
    read_and_plot_history_conf_mat(history_txt, conf_mat_txt)
    exit()

    type = '04'
    normal_path = 'F:/pythonWorksplaces/emonet-master/output/normal/'
    normal_not_path = 'F:/pythonWorksplaces/emonet-master/output/normal_not/'
    normal_emotions = get_data_from_csv(normal_path, type)
    samples,labels = parse_data(normal_emotions,1)
    normal_not_emotions = get_data_from_csv(normal_not_path, type)