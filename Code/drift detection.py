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
from sklearn.metrics import accuracy_score
from collections import Counter



# load data
def load_data(file_path):
    
    data = pd.read_csv(file_path)    
    words = data['x']
    y = data['y'].values
    
    return words, y



def drift_detection(file_path, name):

    words, y = load_data(file_path)
    
    
    # data split
    ini_train_size = 50
    win_size = 50
    
    
    # transfer words into vertors for LDA learning
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
    
    t0 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    t7 = []
    t8 = []
    t9 = []
    t_change = []
    
    for i in range(10):
        topic_stat1.append(np.sum(topic_id == i))
    d1 = topic_stat1
    d1_r = [0,0,0,0,0,0,0,0,0,0]
    
    
    t0.append(0)
    t1.append(1)
    t2.append(2)
    t3.append(3)
    t4.append(4)
    t5.append(5)
    t6.append(6)
    t7.append(7)
    t8.append(8)
    t9.append(9)
    t_change.append(np.argmax(d1))

    
    # k-fold
    kf = KFold(int((cntTf.shape[0] - ini_train_size) / win_size))
    stream = cntTf[ini_train_size:, :]
    y_stream = y[ini_train_size:]
    
    
    n_real_drift = 0
    n_normal = 0
    
    
    # transfer words intp data for model training
    tfidf_v = TfidfVectorizer(max_features = 500)
    x = tfidf_v.fit_transform(words).toarray()
    x_stream = x[ini_train_size:, :]
    
    
    # initial train set (for Bagging)
    x_train = x[0:ini_train_size, :]
    y_train = y[0:ini_train_size]
    
    
    # train decision tree
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    
    # test decision tree model
    y_pred_ini = clf.predict(x_train)


    # calculate the accuracy of decision tree model
    acc_ini = accuracy_score(y_train, y_pred_ini.T)
    
    a = 0
    a_count = []
    drift_idx = []
    drift_count = []
    
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        words_test = stream[test_index, :]
        y_test = y_stream[test_index]
        x_test = x_stream[test_index, :]
        
        
        # get topics for some given samples
        docres2 = lda.transform(words_test)
        topic_id = np.argmax(docres2, axis = 1)
        topic_stat2 = []
        for i in range(10):
            topic_stat2.append(np.sum(topic_id == i))
        d2 = topic_stat2
        d2_r = [np.abs(d2[i] - d1[i]) for i in range(0, len(d2))]
        
        t0.append(0)
        t1.append(1)
        t2.append(2)
        t3.append(3)
        t4.append(4)
        t5.append(5)
        t6.append(6)
        t7.append(7)
        t8.append(8)
        t9.append(9)
        t_change.append(np.argmax(d2))
        
        # K-S test
        statistic, p = stats.ks_2samp(d1_r, d2_r)
        a_count.append(p)
    

        # test decision tree model
        y_pred = clf.predict(x_test)
        
        
        # calculate the accuracy of decision tree model
        acc = accuracy_score(y_test, y_pred.T)
    
        
        # count the number of drift
        if p < 0.1:
            n_real_drift += 1
            drift_count.append(p)
            drift_idx.append(a)
        else:
            n_normal += 1
            
            # retrain the LDA model
            lda = LatentDirichletAllocation(n_components = 10, random_state = 0)
            lda.fit_transform(words_test)
        
        docres1 = docres2
        d1 = d2
        d1_r = d2_r
        acc_ini = acc
        a = a+1
    
    print('-----------------------------------------------------')
    print('n_real_drift:', n_real_drift, 'n_normal:', n_normal)
    # print(a_count)
    print('-----------------------------------------------------')
    
    '''
    plt.figure(figsize=(5.5,2.5))
    plt.scatter(np.arange(len(a_count)), a_count)
    plt.scatter(drift_idx, drift_count, marker = '*', color = 'red', s=80)
    plt.axhline(0.1, linestyle = '--', color = 'lightgreen', label = 'a=0.1, drift trigger')
    plt.xlabel('Time point')
    plt.ylabel('P-value')
    # plt.xlim(0, 25)
    plt.ylim(-0.5, 1.5)
    plt.legend(loc = 'upper right')
    plt.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.2)
    plt.savefig('D:/桌面文件/Access/fig2/' +str(name)+'drift'+'.pdf')
    plt.show()
    '''
    
    
    plt.figure(figsize=(6,3))
    plt.scatter(np.arange(len(t0)), t0, color = 'azure', label = 'topic0')
    plt.scatter(np.arange(len(t1)), t1, color = 'lightcyan', label = 'topic1')
    plt.scatter(np.arange(len(t2)), t2, color = 'paleturquoise', label = 'topic2')
    plt.scatter(np.arange(len(t3)), t3, color = 'powderblue', label = 'topic3')
    plt.scatter(np.arange(len(t4)), t4, color = 'lightblue', label = 'topic4')
    plt.scatter(np.arange(len(t5)), t5, color = 'lightskyblue', label = 'topic5')
    plt.scatter(np.arange(len(t6)), t6, color = 'skyblue', label = 'topic6')
    plt.scatter(np.arange(len(t7)), t7, color = 'lightsteelblue', label = 'topic7')
    plt.scatter(np.arange(len(t8)), t8, color = 'deepskyblue', label = 'topic8')
    plt.scatter(np.arange(len(t9)), t9, color = 'cornflowerblue', label = 'topic9')
    
    plt.plot(t_change, color = 'red', marker = 'o', label = 'topic change')
    
    plt.xlabel('Time point')
    plt.ylabel('Topic index')
    # plt.xlim(0, 25)
    plt.ylim(-1, 15)
    plt.legend(loc = 'upper right', ncol = 4)
    plt.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.2)
    plt.savefig('D:/桌面文件/Access/fig2/' +str(name)+'.pdf')
    plt.show()
    




if __name__ == '__main__':
    
    path = ["D:/桌面文件/Access/amazon评论 - process/评论 process binary/2显示器.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/3手机壳.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/7湿巾.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/11T恤.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/21维生素C.csv",
            "D:/桌面文件/Access/amazon评论 - process/评论 process binary/24泡面.csv"]
    
    # path = ["D:/桌面文件/Access/amazon评论 - process/评论 process binary/2显示器.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/3手机壳.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/7湿巾.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/9液态洗手液.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/11T恤.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/12外套.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/13长裤.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/17n95口罩.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/18护目镜.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/21维生素C.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/23金枪鱼罐头.csv",
    #         "D:/桌面文件/Access/amazon评论 - process/评论 process binary/24泡面.csv"]
    
    
    
    fig_name = ["2","3","7","11","21","24"]
                
    # fig_name = ["2",
    #         "3",
    #         "7",
    #         "9",
    #         "11",
    #         "12",
    #         "13",
    #         "17",
    #         "18",
    #         "21",
    #         "23",
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
        
        
        
        
        
        
        
        
        
        
        
        
