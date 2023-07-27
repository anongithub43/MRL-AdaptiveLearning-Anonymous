# -*- coding: utf-8 -*-
"""
This file contains the functions to generate and perform the MDP clustering
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tqdm import tqdm #progress bar
import binascii
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
#from xgboost import XGBClassifier
#from sklearn.model_selection import GridSearchCV
from collections import Counter
from itertools import groupby
from operator import itemgetter
from testing import plot_features

from testing import R2_value_training, training_value_error, training_accuracy, \
    predict_cluster, R2_value_testing, testing_value_error, testing_accuracy, \
        next_clusters

# suppressing warnings from here because sklearn forces warnings
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#################################################################


#################################################################
# Funtions for Initialization


# split_train_test_by_id() takes in a dataframe of all the data, 
# returns Testing and Training dataset dataframes with the ratio of testing
# data defined by float test_ratio
def split_train_test_by_id(data, # dataframe: all the data
                           test_ratio, # float: portion of data for testing
                           id_column): # str: name of identifying ID column
    
    def test_set_check(identifier, test_ratio):
        return binascii.crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# initializeClusters() takes as input a dataframe,
# a clustering algorithm, a number of clusters n_clusters,
# and a random seed (optional) and returns a dataframe
# with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
def initializeClusters(df,  # pandas dataFrame: MUST contain a "RISK" column
                       clustering='Agglomerative',  # string: clustering algorithm
                       n_clusters= None, # number of clusters
                       distance_threshold= 0.3,
                       random_state=0):  # random seed for the clustering
    df = df.copy()
    if clustering == 'KMeans':
        output = KMeans(
                n_clusters=n_clusters, random_state=random_state).fit(
                        np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Agglomerative':
        output = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold = distance_threshold).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Birch':
        output = Birch(
            n_clusters=n_clusters).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    else:
        output = LabelEncoder().fit_transform(np.array(df.RISK).reshape(-1, 1))
    df['CLUSTER'] = output
    df['CLUSTER'] = df['CLUSTER'].astype(int)
    df['NEXT_CLUSTER'] = df['CLUSTER'].shift(-1)
    df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 0
    df['NEXT_CLUSTER'] = df['NEXT_CLUSTER'].astype(int)
    df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 'None'
    return(df)
#################################################################


#################################################################
# Function for the Iterations

# findConstradiction() takes as input a dataframe and returns the tuple with
# initial cluster and action that have the most number of contradictions or
# (-1, -1) if no such cluster existss
# METHOD FOR FINDING CONTRADICTION:
# METHOD1: Let s,a be a state action pair. We count the number of occurences of 
# different NEXT_CLUSTERS under action a starting from s. Let s' be NEXT_CLUSTER
# most elements from s went to under a. Let n2 be the number of elements that
# didn't go to s' under a. We chose (s,a) to maximize n2.
# METHOD2: Here n2 is chosen as the number of elements that went to the second
# most frequent NEXT_CLUSTER.
def findContradiction(df, # pandas dataFrame
                      th): # integer: threshold split size
    X = df.loc[:, ['CLUSTER', 'NEXT_CLUSTER', 'ACTION']]
    X = X[X.NEXT_CLUSTER != 'None']
    count = X.groupby(['CLUSTER', 'ACTION'])['NEXT_CLUSTER'].nunique()
    contradictions = list(count[list(count > 1)].index)
    
#    #METHOD 1 
#    if len(contradictions) > 0:
#        ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
#                'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
#            sort_values(ascending=False).ravel())[1:]) for i in contradictions]
    
    #METHOD 2
    if len(contradictions) > 0:
        ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
                'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
            sort_values(ascending=False).ravel())[1:2]) for i in contradictions]
        # print('ncontradictions: ', ncontradictions)
        if max(ncontradictions) > th:
            selectedCont = contradictions[ncontradictions.index(
                    max(ncontradictions))]
            return(selectedCont)
    
    return((-1, -1))


# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
def contradiction(df,  # pandas dataFrame
                  i,  # integer: initial clusters
                  a):  # integer: action taken
    nc = list(df.query('CLUSTER == @i').query(
            'ACTION == @a').query('NEXT_CLUSTER != "None"')['NEXT_CLUSTER'])
    # print('nc: ', nc)
    if len(nc) == 1:
        return (None, None)
    else:
        return a, multimode(nc)[0]


# multimode() returns a list of the most frequently occurring values. 
# Will return more than one result if there are multiple modes 
# or an empty list if *data* is empty.
def multimode(data):
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))


# split() takes as input a dataframe, an initial cluster, an action, a target
# cluster that is a contradiction c, then number of features,
# and an iterator k (that is the indexer of the next cluster), as well as the
# predictive classification algorithm used
# Returns a new dataframe with the contradiction resolved, and the best fit score
# for the splitting model (if GridSearch used)
def split(df,  # pandas dataFrame
          i,  # integer: initial cluster
          a,  # integer: action taken
          c,  # integer: target cluster
          pfeatures,  # integer: number of features
          k,  # integer: indexer for next cluster
          classification='LogisticRegression', # string: classification aglo
          split_classifier_params={'random_state':0}): # dict: of classifier params

    g1 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] == c)]
    g2 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] != c) & (
                    df['NEXT_CLUSTER'] != 'None')]
    g3 = df[(df['CLUSTER'] == i) & (
            ((df['ACTION'] == a) & (df['NEXT_CLUSTER'] == 'None')) | (
                    df['ACTION'] != a))]
    # print('g1: ', g1)
    # print('g2: ', g2)
    # print('g3: ', g3[g3['FEATURE_0'] > 4][g3['FEATURE_1'] < -3][g3['FEATURE_1'] > -4])

    # print('g1 shape: ', g1.shape)
    # print('g2 shape: ', g2.shape)
    # print('g3 shape: ', g3.shape)

    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):

        d = pd.DataFrame(groups[j].iloc[:, 2:2+pfeatures].values.tolist())

        data[j] = d

    data[0].insert(data[0].shape[1], "GROUP", np.zeros(data[0].shape[0]))
    data[1].insert(data[1].shape[1], "GROUP", np.ones(data[1].shape[0]))

    training = pd.concat([data[0], data[1]])

    tr_X = training.iloc[:, :-1]
    tr_y = training.iloc[:, -1:]

    if classification == 'LogisticRegression':
        m = LogisticRegression(**split_classifier_params)
    elif classification == 'LogisticRegressionCV':
        m = LogisticRegressionCV(**split_classifier_params)
    elif classification == 'DecisionTreeClassifier':
        m = DecisionTreeClassifier(**split_classifier_params)
        # params = {
        # 'max_depth': [3, None]
        # }
        # m = GridSearchCV(m, params,cv = 5)
    elif classification == 'RandomForestClassifier':
        m = RandomForestClassifier(**split_classifier_params)
        # params = {
        # 'max_depth': [3, None]
        # }
        # m = GridSearchCV(m, params,cv = 5)
    #elif classification == 'XGBClassifier':
        #m = XGBClassifier()        
    elif classification == 'MLPClassifier':
        m = MLPClassifier(**split_classifier_params)
    elif classification == 'AdaBoostClassifier':
        m = AdaBoostClassifier(**split_classifier_params)
    else:
        m = LogisticRegression(**split_classifier_params)
        
    m.fit(tr_X, tr_y.values.ravel())
    try:
        score = m.best_score_
    except:
        score = None


    ids = g2.index.values

    test_X = data[2]

    if len(test_X) != 0:
        Y = m.predict(test_X)
        g3.insert(g3.shape[1], "GROUP", Y.ravel())
        id2 = g3.loc[g3["GROUP"] == 1].index.values
        ids = np.concatenate((ids, id2))
    
    
    df.loc[df.index.isin(ids), 'CLUSTER'] = k
    newids = ids-1
    df.loc[(df.index.isin(newids)) & 
           (df['ID']== df['ID'].shift(-1)), 'NEXT_CLUSTER'] = k

    return df, score


# splitter() is the wrap-up function. Takes the below parameters and
# performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe, as well as incoherences, errors,
# and the dataframe with the optimal split
def splitter(df,  # pandas dataFrame
             pfeatures,  # integer: number of features
             th, # integer: threshold for minimum split
             eta = 25, # incoherence threshold for splits 
             precision_thresh = 1e-14, # precision threshold when considering new min value error
             df_test = None, # df_test provided for cross validation
             testing = False, # True if we are cross validating
             max_k = 6, # int: max number of clusters
             classification='LogisticRegression',  # string: classification alg
             split_classifier_params = {'random_state':0}, # dict: classification params
             h=5,
             gamma=1,
             verbose = False,
             n=-1,
             plot = False):  
    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    training_acc = []
    testing_acc = []
    testing_error = []
    training_error = []
    
    incoherences = []
    split_scores = []
    thresholds = []
    
    # determine if the problem has OG cluster
    if 'OG_CLUSTER' in df.columns:
        grid = True
    else:
        grid = False

    
    k = df['CLUSTER'].nunique() #initial number of clusters 
    nc = k #number of clusters
    
    df_new = deepcopy(df)
    
    # storing optimal df
    best_df = None
    opt_k = None
    min_error = float('inf')
    
    # backup values in case threshold fails
    backup_min_error = float('inf')
    backup_df = None
    backup_opt_k = None
    
    # Setting progress bar--------------
    split_bar = tqdm(range(max_k-k))
    split_bar.set_description("Splitting...")
    # Setting progress bar--------------
    for i in split_bar:
        split_bar.set_description("Splitting... |#Clusters:%s" %(nc))
        cont = False
        c, a = findContradiction(df_new, th)
        if c != -1:
            
            # finding contradictions and splitting
            a, b = contradiction(df_new, c, a)

            #print('a, b: ', a, b)
            
            if verbose:
                print('Cluster splitted', c,'| Action causing contradiction:', a, '| Cluster most elements went to:', b)
            df_new, score = split(df_new, c, a, b, pfeatures, nc, classification,split_classifier_params)
            split_scores.append(score)

            #print(df_new['CLUSTER'].unique())
            #plot_features(df_new, 'FEATURE_0', 'FEATURE_1')
            
            # calculate incoherences
            next_clus = next_clusters(df_new)
            next_clus['incoherence'] = (1-next_clus['purity'])*next_clus['count']
            next_clus.reset_index(inplace=True)
            next_clus = next_clus.groupby('CLUSTER').sum(numeric_only=True)
            max_inc = next_clus['incoherence'].max()
            incoherences.append(max_inc)
                
            
            # error and accuracy calculations
            
            R2_train = R2_value_training(df_new)
            training_R2.append(R2_train)       
            train_error = training_value_error(df_new, gamma, relative=False, h=h)
            training_error.append(train_error)            
            
            if grid:
                train_acc = training_accuracy(df_new)[0]
                training_acc.append(train_acc)                        
            
            
            if testing:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
                testing_R2.append(R2_test)
                test_error = testing_value_error(df_test, df_new, model, pfeatures, gamma, relative=False,h=h)
                testing_error.append(test_error)
                
                if grid:
                    test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
                    testing_acc.append(test_acc)
                
                
            # printing error and accuracy values
            if verbose:
                print('training value R2:', R2_train)
                print('training value error:', train_error)
                if grid:
                    print('training accuracy:', train_acc)
                if testing:
                    print('testing value R2:', R2_test)
                    print('testing value error:', test_error)
                    if grid:
                        print('testing accuracy:', test_acc)
            #print('predictions:', get_predictions(df_new))
            #print(df_new.head())
            
            # update optimal dataframe if inc threshold and min error met
            # threshold calculated using eta * sqrt(number of datapoints) / 
            # number of clusters
            threshold = eta*df_new.shape[0]**0.5/(nc+1)
            thresholds.append(threshold)
            if verbose:
                print('threshold:', threshold, 'max_incoherence:', max_inc)
            
            # only update the best dataframe if training error is smaller
            # than previous training error by at least precision_thresh, 
            # and also if maximum incoherence is lower than calculated threshold
            #print('max inc, threshold: ', max_inc, threshold)
            #print('train error: ', train_error, min_error - precision_thresh)
            if max_inc < threshold and train_error < (min_error-precision_thresh): 
                min_error = train_error
                best_df = df_new.copy()
                opt_k = nc+1
                if verbose:
                    print('new opt_k', opt_k)
            
            # code for storing optimal clustering even if incorrect incoherence 
            # threshold is chosen and nothing passes threshold; to prevent 
            # training interruption
            elif opt_k == None and train_error < (backup_min_error-precision_thresh):
                backup_min_error = train_error
                backup_df = df_new.copy()
                backup_opt_k = nc+1
            
            cont = True
            nc += 1
        if not cont:
            break
        if nc >= max_k:
            if verbose:
                print('Optimal # of clusters reached')
            break
        
        
        # plot every 20 iterations
        
        # if plot:
        #     if i%20 == 0: 
        #         its = np.arange(k+1, nc+1)
        #         fig2, ax2 = plt.subplots()
        #         ax2.plot(its, training_error, label = "Training Error")
        #         if testing:
        #             ax2.plot(its, testing_error, label = "Testing Error")
        #         if n>0:
        #             ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        #         ax2.set_ylim(0)
        #         ax2.set_xlabel('# of Clusters')
        #         ax2.set_ylabel('Value error')
        #         ax2.set_title('Value error by number of clusters')
        #         ax2.legend()
        #         plt.show()
        
        
    # in the case that threshold prevents any values from passing, use backup
    if opt_k == None: 
        opt_k = backup_opt_k
        best_df = backup_df
        min_error = backup_min_error
            
    
    # plotting functions
    ## Plotting accuracy and value R2
    its = np.arange(k+1, nc+1)
    if plot:
        if grid:
            fig1, ax1 = plt.subplots()
            #ax1.plot(its, training_R2, label= "Training R2")
            ax1.plot(its, training_acc, label = "Training Accuracy")
            if testing:
                ax1.plot(its, testing_acc, label = "Testing Accuracy")
                #ax1.plot(its, testing_R2, label = "Testing R2")
            if n>0:
                ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
            ax1.set_ylim(0,1)
            ax1.set_xlabel('# of Clusters')
            ax1.set_ylabel('R2 or Accuracy %')
            ax1.set_title('R2 and Accuracy During Splitting')
            ax1.legend()
        ## Plotting value error E((v_est - v_true)^2)
        fig2, ax2 = plt.subplots()
        norm_max = max(incoherences)
        ax2.plot(its, training_error, label = "Training Error")
        ax2.plot(its, np.array(incoherences)/norm_max, label = "Max Incoherence")
        ax2.plot(its, np.array(thresholds)/norm_max, 'r-', label = "Threshold")
        if testing:
            ax2.plot(its, testing_error, label = "Testing Error")
        if n>0:
            ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel('# of Clusters')
        ax2.set_ylabel('Value error')
        ax2.set_title('Value error by number of clusters')
        ax2.legend()
        plt.show()
    
    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns = ['Clusters', 'Error'])
    df_incoherences = pd.DataFrame(list(zip(its, incoherences)), \
                                  columns = ['Clusters', 'Incoherences'])
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                  columns = ['Clusters', 'Error'])
        return (df_new, df_incoherences, df_train_error,df_test_error, best_df, opt_k, split_scores)
    return(df_new, df_incoherences, df_train_error,testing_error, best_df, opt_k, split_scores)

#################################################################



# Splitter algorithm with Group K-fold cross-validation (number of folds from param cv)
# Returns dataframes of incoherences, errors, and splitter split-scores; these
# can be used to determine optimal clustering. 
def fit_CV(df,
          pfeatures,
          th,
          clustering,
          distance_threshold,
          eta, 
          precision_thresh,
          classification,
          split_classifier_params,
          max_k,
          n_clusters,
          random_state,
          h,
          gamma=1,
          verbose = False,
          cv=5,
          n=-1,
          plot = False):
    
    
    df_training_error = pd.DataFrame(columns=['Clusters'])
    df_testing_error = pd.DataFrame(columns=['Clusters'])
    df_incoherences = pd.DataFrame(columns=['Clusters'])
    
    gkf = GroupKFold(n_splits=cv)
    # shuffle the ID's (create a new column), and do splits based on new ID's
    random.seed(datetime.now())
    g = [df for _, df in df.groupby('ID')]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(['ID'], sort=False).ngroup()
    df['ID_shuffle'] = ids
    
    for train_idx, test_idx in gkf.split(df, y=None, groups=df['ID_shuffle']):

        df_train = df[df.index.isin(train_idx)]
        df_test = df[df.index.isin(test_idx)]
        #print('IDs in testing', df_test['ID'].unique())
        #################################################################
        # Initialize Clusters
        df_init = initializeClusters(df_train,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)
        #k = df_init['CLUSTER'].nunique()
        #print('k', k)
        #print(df_init)
        #################################################################
        
        #################################################################
        # Run Iterative Learning Algorithm
        
        df_new,incoherences,training_error,testing_error, best_df, opt_k, split_scores = splitter(df_init,
                                          pfeatures,
                                          th,
                                          eta = eta,
                                          precision_thresh = precision_thresh,
                                          df_test = df_test,
                                          testing = True,
                                          max_k = max_k,
                                          classification = classification,
                                          split_classifier_params = split_classifier_params,
                                          h = h, 
                                          gamma = gamma,
                                          verbose = False,
                                          n=n,
                                          plot = plot)
        
        df_training_error = df_training_error.merge(training_error, \
                                                    how='outer', on=['Clusters'])
        df_testing_error = df_testing_error.merge(testing_error, \
                                                  how='outer', on=['Clusters'])
        df_incoherences = df_incoherences.merge(incoherences, \
                                                how='outer', on=['Clusters'])
    
    df_training_error.set_index('Clusters', inplace=True)
    df_testing_error.set_index('Clusters', inplace=True)
    df_incoherences.set_index('Clusters', inplace=True)
    
    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    df_incoherences.dropna(inplace=True)
    
    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    cv_incoherences = np.mean(df_incoherences, axis=1)

    
    
    if plot:
        fig1, ax1 = plt.subplots()
        #its = np.arange(k+1,k+1+len(cv_training_error))
        ax1.plot(cv_training_error.index.values, cv_training_error, label= "CV Training Error")
        #ax1.plot(its, cv_testing_error, label = "CV Testing Error")
        ax1.plot(cv_testing_error.index.values, cv_testing_error, label= "CV Testing Error")
        #ax1.plot(its, training_acc, label = "Training Accuracy")
        #ax1.plot(its, testing_acc, label = "Testing Accuracy")
        if n>0:
            ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel('# of Clusters')
        ax1.set_ylabel('Mean CV Error or Accuracy %')
        ax1.set_title('Mean CV Error and Accuracy During Splitting')
        ax1.legend()
    
    return (cv_incoherences, cv_training_error,cv_testing_error, split_scores)