# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:41:25 2022

@author: Kun Wang
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter



# load data
def load_data(file_path):
    
    data = pd.read_csv(file_path)    
    words = data['x']
    y = data['y'].values
    
    return words, y



def drift_detection(file_path, name):

    words, y = load_data(file_path)
    np.random.seed(0)
    
    # data split
    ini_train_size = 50
    win_size = 50
    
    
    # transfer words into vectors for LDA learning
    vectorizer = CountVectorizer(max_features = 500)
    cntTf = vectorizer.fit_transform(words)
    # print (cntTf.toarray())
    
    
    # initial train set
    words_train = cntTf[0:ini_train_size, :]
    y_train = y[0:ini_train_size]
    
    
    # load LDA model
    lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
    docres1 = lda.fit_transform(words_train)
    topic_id = np.argmax(docres1, axis = 1)
    topic_stat1 = []
    
    
    for i in range(10):
        topic_stat1.append(np.sum(topic_id == i))
    d1 = topic_stat1
    d1_r = [0,0,0,0,0,0,0,0,0,0]

    
    # k-fold
    kf = KFold(int((cntTf.shape[0] - ini_train_size) / win_size))
    stream = cntTf[ini_train_size:, :]
    y_stream = y[ini_train_size:]
    
    
    n_real_drift = 0
    n_normal = 0
    
    
    # transfer words into data for model training
    tfidf_v = TfidfVectorizer(max_features = 500)
    x = tfidf_v.fit_transform(words).toarray()
    x_stream = x[ini_train_size:, :]
    
    
    # initial train set (for Bagging)
    x_train = x[0:ini_train_size, :]
    y_train = y[0:ini_train_size]
    
    
    # build a set of decision trees
    Bagging = []
    for i in range(10):
        idx = np.where(topic_id == i)
        x_train_subset = x_train[idx]
        y_train_subset = y_train[idx]
            
        clf = DecisionTreeClassifier(random_state = 0)
        clf.fit(x_train_subset, y_train_subset)
        
        Bagging.append(clf)

    
    # train decision tree
    clf_single = DecisionTreeClassifier(random_state = 0)
    clf_single.fit(x_train, y_train)

    
    a_count = []
    acc_chunk = []
    acc_chunk_bagging = []
    acc_final = []
    
    pred_final = np.empty(0)
    
    
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        words_test = stream[test_index, :]
        y_test = y_stream[test_index]
        x_test = x_stream[test_index, :]
        
        
        # get topics for some given samples
        docres2 = lda.transform(words_test)
        topic_id2 = np.argmax(docres2, axis = 1)
        topic_stat2 = []
        for i in range(10):
            topic_stat2.append(np.sum(topic_id2 == i))
        d2 = topic_stat2
        d2_r = [np.abs(d2[i] - d1[i]) for i in range(0, len(d2))]

        
        # K-S test
        statistic, p = stats.ks_2samp(d1_r, d2_r)
        a_count.append(p)
        
        
        # test the bagging model
        pred_bag = np.zeros((x_test.shape[0]))
        
        for i in range(10):
            
            idx = np.where(topic_id == i)
            x_test_subset = x_test[idx] 
            y_test_subset = y_test[idx] 
            
            pred_bag[idx] = Bagging[i].predict(x_test_subset)
            
            # retrain the base learner
            clf = DecisionTreeClassifier(random_state = 0)
            clf.fit(x_test_subset, y_test_subset)
            Bagging[i] = clf
            

        # test decision tree model
        y_pred = clf_single.predict(x_test)
        
        
        # calculate the accuracy of decision tree model
        acc_bagging = accuracy_score(y_test, pred_bag.T)
        acc_chunk_bagging.append(acc_bagging)
        
        acc = accuracy_score(y_test, y_pred.T)
        acc_chunk.append(acc)
        
        # if acc_bagging <= acc:
        #     acc_final.append(acc)
        #     pred_final = np.hstack((pred_final, y_pred)) 
            
        # else:
        #     acc_final.append(acc_bagging)
        pred_final = np.hstack((pred_final, pred_bag))
        
    
        # count the number of drift
        if p < 0.1:
            
            n_real_drift += 1
            
            # # retrain decision tree
            # clf_single = DecisionTreeClassifier(random_state = 0)
            # clf_single.fit(x_test, y_test)
    
        else:
            n_normal += 1
            
            
        # retrain the LDA model
        lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
        lda.fit_transform(words_test)
        
        
        docres1 = docres2
        d1 = d2
        d1_r = d2_r
        acc_ini = acc
        
    
    print('-----------------------------------------------------')
    print('n_real_drift:', n_real_drift, 'n_normal:', n_normal)
    print('accuracy:', accuracy_score(y[ini_train_size:], pred_final))
    print('f1-score:', f1_score(y[ini_train_size:], pred_final, average='macro'))
    # print('acc_average:', np.mean(acc_chunk))
    # print('acc_average_bagging:', np.mean(acc_chunk_bagging))
    # print('acc_average_final:', np.mean(acc_final))
    # print(a_count)
    print('-----------------------------------------------------')
    
    
    


if __name__ == '__main__':
    
    # path = ["D:/桌面文件/Access/amazon评论 - process/评论 process binary/3手机壳.csv"]
    
    path = ["D:/桌面文件/Access/amazon评论 - process/评论 process binary/2显示器.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/3手机壳.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/7湿巾.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/11T恤.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/13长裤.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/21维生素C.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/24泡面.csv"]
    
    # path = ["D:/桌面文件/Access/amazon评论 - process/评论 process binary/2显示器.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/3手机壳.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/7湿巾.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/11T恤.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/13长裤.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/17n95口罩.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/18护目镜.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/21维生素C.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/24泡面.csv"]
    
    
    
    # fig_name = ["2"]
    
    fig_name = ["2",
            "3",
            "7",
            "11",
            "13",
            "21",
            "24"]
                
    # fig_name = ["2",
    #         "3",
    #         "7",
    #         "11",
    #         "13",
    #         "17",
    #         "18",
    #         "21",
    #         "24"]
    
    
    
    # path = ["D:/桌面文件/ECR/amazon评论 - process/评论 process binary/1无线耳机.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/2显示器.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/3手机.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/4打印机.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/5键盘.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/6厕纸.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/7湿巾.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/8免洗洗手液.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/9液态洗手液.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/10清洁喷雾.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/11T恤.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/12外套.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/13长裤.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/14连衣裙.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/15运动鞋.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/16口罩.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/17n95口罩.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/18护目镜.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/19面罩.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/20医用手套.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/21维生素C.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/22巧克力.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/23金枪鱼罐头.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/24泡面.csv",
    #         "D:/桌面文件/ECR/amazon评论 - process/评论 process binary/25咖啡豆.csv"]
    
    
    for i in range(len(path)):
        print(path[i])
        drift_detection(path[i], fig_name[i])
        
        
        
        
        
        
        
        
        
        
        
        
