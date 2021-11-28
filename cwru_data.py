#-*- coding:utf-8 -*-
import os,math
import torch
import errno
import random
import numpy as np
from scipy.io import loadmat


class CWRU:

    def __init__(self, exp, rpm, length):
        if exp not in ('12k_Drive_End_Fault', '12k_Fan_End_Fault', '48k_Drive_End_Fault'):
            print("wrong experiment name: {}".format(exp))
            exit(1)
        if rpm not in ('0', '1', '2', '3'):
            print("wrong rpm value: {}".format(rpm))
            exit(1)
        '''
        标号  转速
        0	 1797
        1	 1772
        2	 1750
        3	 1730
        '''
        # root directory of all data
        rdir = os.path.join('./data/CWRU')

        all_file = os.listdir(rdir)
        self.rpm = rpm
        self.length = length  # sequence length
        self._load_and_slice_data(rdir,all_file, exp)
        # shuffle training and test arrays
        self._shuffle()
        

    def _extract_error_list(self,all_file,inputs,rdir):
        error_list = [] # 抽取出输入的要检测的数据
        for name in enumerate(all_file):
            # directory of this file
            information = name[1].split('_')
            if information[0]==inputs[0] and information[1]==inputs[1] and information[2]==inputs[2] and information[4]==self.rpm:
                fpath = rdir+'/'+name[1]
                error_list.append(fpath)
        self.nclasses = len(error_list)
        return error_list
    
        
    def _accelarate_data(self,mat_dict,BADEFE):
        '''
        提取加速度数据
        '''
        key = list(filter(lambda x: BADEFE in x, list(mat_dict.keys())))[0]
        # filter()函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
        time_series = mat_dict[key][:, 0]   # 每个采样点一个数据
        idx_last = -(time_series.shape[0] % self.length)    # 去除余数
        clips = time_series[:idx_last].reshape(-1, self.length)
        n = clips.shape[0]  # 用self.length表示一段时间内的数据
        # n为数据的行数,self.length为数据的列数
        n_split = math.floor(3 * n / 4)
        return clips,n_split
        
    
    def _load_and_slice_data(self, rdir,all_file, infos):
        '''
        DE - drive end accelerometer data    驱动端加速度数据
        FE - fan end accelerometer data      风扇端加速度数据
        BA - base accelerometer data        基座加速度数据（正常）
        time - time series data               时间序列数据
        RPM- rpm during testing        每秒钟多少转，除以60为旋转频率
        '''
        X_train_DE = np.zeros((0, self.length))
        X_test_DE = np.zeros((0, self.length))
        X_train_BA = np.zeros((0, self.length))
        X_test_BA = np.zeros((0, self.length))
        X_train_FE = np.zeros((0, self.length))
        X_test_FE = np.zeros((0, self.length))
        
        self.y_train = []
        self.y_test = []
        
        inputs = infos.split('_')
        error_list = self._extract_error_list(all_file,inputs,rdir)
          
        for idx, fpath in enumerate(error_list):
            mat_dict = loadmat(fpath)
            clips,n_split = self._accelarate_data(mat_dict,"DE")
            X_train_DE = np.vstack((X_train_DE, clips[:n_split]))
            X_test_DE = np.vstack((X_test_DE, clips[n_split:]))
            self.y_train += [idx] * n_split
            self.y_test += [idx] * (clips.shape[0] - n_split)
            
            clips,n_split = self._accelarate_data(mat_dict,"BA")
            X_train_BA = np.vstack((X_train_BA, clips[:n_split]))
            X_test_BA = np.vstack((X_test_BA, clips[n_split:]))
            
            clips,n_split = self._accelarate_data(mat_dict,"FE")
            X_train_FE = np.vstack((X_train_FE, clips[:n_split]))
            X_test_FE = np.vstack((X_test_FE, clips[n_split:]))
            
        self.X_train= np.zeros((X_train_DE.shape[0], self.length,3))
        self.X_test= np.zeros((X_test_DE.shape[0], self.length,3))
        self.X_train[:,:,0] = X_train_DE
        self.X_train[:,:,1] = X_train_BA
        self.X_train[:,:,2] = X_train_FE
        self.X_test[:,:,0] = X_test_DE
        self.X_test[:,:,1] = X_test_BA
        self.X_test[:,:,2] = X_test_FE

    
    def _one_hot(self,x, class_count):
    	# 第一构造一个[class_count, class_count]的对角线为1的向量
    	# 第二保留label对应的行并返回
    	return np.array(torch.eye(class_count)[x,:])


    def _shuffle(self):
        # shuffle training samples,y_train is one-hot mode
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index,:,:]
        self.y_train = list(self.y_train[i] for i in index)
        self.y_train=self._one_hot(self.y_train,self.nclasses)

        # shuffle test samples,y_test is one-hot mode
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index,:,:]
        self.y_test = list(self.y_test[i] for i in index)
        self.y_test=self._one_hot(self.y_test,self.nclasses)
        

mydata = CWRU("12k_Drive_End_Fault", '0', 256)