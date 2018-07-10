
from nltk.util import ngrams
import sys
import pickle
from sklearn.utils import resample
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score,recall_score,f1_score

from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline,make_pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.datasets import  make_classification
from sklearn.ensemble import RandomForestClassifier

import nltk
import string
import numpy as np
from subprocess import call
from pathlib import Path
import time
import os
import operator


def parse(text):
    text = text.lower()
    # text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text


def scores(y_true, y_pred, labels,verbose):
    formatted_y_true= []
    for r in y_true:
        formatted_y_true.append( [ (1 if r==l else 0) for l in labels])
    formatted_y_pred = []
    for r in y_pred:
        formatted_y_pred.append([(1 if r == l else 0) for l in labels])
    formatted_y_true= np.array(formatted_y_true)
    formatted_y_pred = np.array(formatted_y_pred)
    accuracy_scores=[]
    precision_scores=[]
    recall_scores=[]
    f1_scores = []

    for i in range(formatted_y_true.shape[1]):
        accuracy_scores.append(accuracy_score(formatted_y_true[:, i], formatted_y_pred[:, i]))
        precision_scores.append(precision_score(formatted_y_true[:, i], formatted_y_pred[:, i]))
        recall_scores.append(recall_score(formatted_y_true[:, i], formatted_y_pred[:, i]))
        f1_scores.append(f1_score(formatted_y_true[:, i], formatted_y_pred[:, i]))
        if verbose>0:
            print("Col {} ".format(i), labels[i])
            print('confusion_matrix ', confusion_matrix(formatted_y_true[:, i], formatted_y_pred[:, i]))
            print('accurracy ', "{:10.2f}".format(accuracy_scores[-1]))
            print('precision ', "{:10.2f}".format(precision_scores[-1]))
            print('recall ', "{:10.2f}".format(recall_scores[-1]))
            print('f1 ', "{:10.2f}".format(f1_scores[-1]))
            print("")
    return accuracy_scores,precision_scores,recall_scores,f1_scores


F={}

class Ngram(object):
    def __init__(self, classifierName, dataset, generator, n, remove, balancing, cluster):
        freq_dist = []
        self.classifierName=classifierName
        self._clf=None
        self.classifier=None
        self.generator=generator
        self.all_ngrams = []
        self.dataset = dataset
        self.tokenizedDataset = {}
        self.tokenizedDatasetInNgrams={}
        self.balancing = balancing
        self.cluster=cluster
        self.allLabels= sorted(list(set(dataset['label'])))
        self.n = n
        self.predictProbs=[]

        for index, data in dataset.iterrows():
            if((int(index)%1000)==0):
                print('processing...',str(index))

            data=self.text2Ngrams(data.text)
            self.tokenizedDataset[index]=data
            freq_dist.extend(data)

        freq_dist = nltk.FreqDist(freq_dist)
        ngramsToRemoved = [word for word in freq_dist if (word in string.punctuation) or (not word.startswith('***') and freq_dist[word] <= remove)]
        list_all_ngrams = list(freq_dist.keys())
        for ngram in ngramsToRemoved: list_all_ngrams.remove(ngram)
        self.all_ngrams.extend(list_all_ngrams)
        self.all_ngrams_index = {k: v for v, k in enumerate(self.all_ngrams)}
        for index, data in dataset.iterrows():
            self.tokenizedDatasetInNgrams[index]= set(ngram for ngram in self.all_ngrams if ngram in self.tokenizedDataset[index])


    def tokenize(self,text):
        # word_tokenize
        return text.lower().split()

    def featurize(self, text,index):
        if(self.classifier=='M'):
            feature = dict([(ngram, True) for ngram in self.tokenizedDatasetInNgrams[index]])
        else:
            feature = ((self.all_ngrams_index[ngram], (ngram in self.tokenizedDatasetInNgrams[index])) for ngram in self.all_ngrams)

        return feature

    # def featurizeMaxEnt(self ,text,index):
    #     feature = dict([(self.all_ngrams_index[ngram], True) for ngram in self.tokenizedDatasetInNgrams[index]])

    def featurize1(self, text,index):
        feature = dict(( ngram , (ngram in self.tokenizedDatasetInNgrams[index])) for ngram in
                       self.all_ngrams)
        return feature

    def featurize2(self, text):
        data = self.text2Ngrams(text)
        feature = dict((ngram, (ngram in data)) for ngram in self.all_ngrams)
        return feature

    def text2Ngrams(self,text):
        tokenized = self.tokenize(text)
        listNgrams = set()
        for i in self.n:
            listNgrams = listNgrams.union(set(ngrams(tokenized, i)))
        data = [' '.join(ngram) for ngram in listNgrams]
        return data


    def getAllNgrams(self):
        return self.all_ngrams

    def extractFeatures(self):
        for index, data in self.dataset.iterrows():
            for ngram in parse(data.text).split():
                self.train_set[ngram][index] = True

    def getTrainset(self):
        return self.train_set

    def balance(self,dataDF,type):
        lengths = []
        dataDFMap ={}

        for l in self.allLabels:
            dataDFMap[l]=list(c for c in dataDF if c[1]==l)

        if(type==1):
            maxCount = max(len(dataDFMap[l]) for l in self.allLabels)
            for l in self.allLabels:
                if (0< len(dataDFMap[l]) < maxCount):
                    dataDF = dataDF + resample(dataDFMap[l],
                                                         replace=True,  # sample with replacement
                                                         n_samples=maxCount - len(dataDFMap[l]),
                                                         # to match majority class
                                                         random_state=1) # reproducible results
        elif(type==2):
            minCount= min(len(dataDFMap[l]) for l in self.allLabels)
            for l in self.allLabels:
                if (0 < len(dataDFMap[l]) > minCount):
                    dataDF = dataDF + resample(dataDFMap[l],
                                               replace=True,  # sample with replacement
                                               n_samples= len(dataDFMap[l]) - minCount,
                                               # to match majority class
                                               random_state=1)  # reproducible results


        elif (type == 3):
            X= np.asarray([list(t[0].values()) for t in dataDF],dtype=bool)
            Y= np.asarray([t[1] for t in dataDF])
            pipe = imbPipeline([
                ('RandomOverSampler', RandomOverSampler(random_state=0))
                               ])
            X_resampled, y_resampled =pipe.fit_sample(X, Y)
            dataDF_resampled =[]
            feature_names = list(dataDF[0][0].keys())
            for i,x in enumerate(X_resampled):
                dataDF_resampled.append((dict(zip(feature_names, x)),y_resampled[i]))
            dataDF=  dataDF_resampled
        else:
            X = np.asarray([list(t[0].values()) for t in dataDF], dtype=bool)
            Y = np.asarray([t[1] for t in dataDF])
            pipe = imbPipeline([
                ('RandomOverSampler', RandomOverSampler(random_state=0))
                , ('smote', SMOTE(kind='svm'))
            ])
            X_resampled, y_resampled = pipe.fit_sample(X, Y)
            dataDF_resampled = []
            feature_names = list(dataDF[0][0].keys())
            for i, x in enumerate(X_resampled):
                dataDF_resampled.append((dict(zip(feature_names, x)), y_resampled[i]))
            dataDF = dataDF_resampled

        for l in self.allLabels:
                lengths.append(len(dataDFMap[l]))

        return dataDF

    def create_dataset(self,n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3,
                       class_sep=0.8, n_clusters=1):
        return make_classification(n_samples=n_samples, n_features=2,
                                   n_informative=2, n_redundant=0, n_repeated=0,
                                   n_classes=n_classes,
                                   n_clusters_per_class=n_clusters,
                                   weights=list(weights),
                                   class_sep=class_sep, random_state=0)

    def train(self,train_data):
        if self.classifierName == "N":
            name = "NaiveBayes"
            self.classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
        elif self.classifierName == "L":
            name = "LogisticRegression"
            self._clf = LogisticRegression()
            self.classifier =  nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "S":
            name = "SVC"
            self._clf =  SVC(class_weight='balanced',probability=True,verbose=False)
            self.classifier = nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "LS":
            name = "LinearSVC"
            self._clf = LinearSVC(penalty="l2")
            self.classifier = nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "P":
            name = "Pipleine"
            self._clf = Pipeline([
                ('classification',OneVsRestClassifier(SVC(class_weight='balanced',probability=True,verbose=False)))])
            self.classifier = nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "P-Select":
            name = "Pipleine"
            self._clf = Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,  tol=1e-3))),
                ('classification',OneVsRestClassifier(SVC(class_weight='balanced',probability=True,verbose=False)))])
            self.classifier = nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "P-KBest":
            name = "Pipleine"
            self._clf = Pipeline([
                ('feature_selection', SelectKBest(chi2)),
                ('classification',OneVsRestClassifier(SVC(class_weight='balanced',probability=True,verbose=False)))])
            self.classifier = nltk.classify.SklearnClassifier(self._clf).train(train_data)
        elif self.classifierName == "RF":
            vectorizer = DictVectorizer(dtype=float, sparse=False)
            X = [vectorizer.fit_transform(dict(d.items()))[0] for (d, l) in train_data]
            Y = [l for (d, l) in train_data]
            self.classifier = RandomForestClassifier(max_depth=None, random_state=1).fit(X,Y)

        # elif self.classifierName == "LS":
        #     name = "LIBLINEAR"
        #     y=[self.allLabels.index(l[1]) for l in train_data]
        #     x=[l[0] for l in train_data]
        #     prob = problem(y,x)
        #     param = parameter('-c 4 -s 2')
        #
        #     self.classifier = liblinear.train(prob, param) # m is a ctype pointer to a model
            # # Convert a Python-format instance to feature_nodearray, a ctypes structure
            # x0, max_idx = gen_feature_nodearray({1:1, 3:1})
            # label = liblinear.predict(m, x0)
        elif self.classifierName == "M":
            # name = "Maxent"
            self.classifier = nltk.maxent.MaxentClassifier.train(train_data)

        return self.classifier

    def classify_many(self,test_data):
        if self.classifierName == 'S' or self.classifierName.startswith('P'):
            vectorizer=DictVectorizer(dtype=float, sparse=True)

            for test in test_data:
                self.predictProbs.append(self._clf.predict_proba(vectorizer.fit_transform(test[0])))

            return self.classifier.classify_many(dict(d.items()) for (d, l) in test_data)
        # elif self.classifierName == 'LS':
        #     labels=[]
        #     for i,d in enumerate(test_data):
        #         x0, max_idx = gen_feature_nodearray(d[0])
        #         label = liblinear.predict(self.classifier, x0)
        #         labels.append(self.allLabels[int(label)])
        #     return labels
        elif self.classifierName == 'RF':
            vectorizer=DictVectorizer(dtype=float, sparse=False)
            X = [vectorizer.fit_transform(dict(d.items()))[0] for (d, l) in test_data]
            Y = [l for (d, l) in test_data]
            self.predictProbs=self.classifier.predict_proba(X)
            return self.classifier.predict(X)

        else:

            return self.classifier.classify_many(dict(d.items()) for (d, l) in test_data)

    def classify_predict(self,features):

        classified = self.classifier.prob_classify_many([dict(features.items())])
        sorted_classified = sorted(classified[0]._prob_dict.items(), key=operator.itemgetter(1), reverse=True)
        return (sorted_classified[0:3])

    def cross_validate(self,    num_folds,verbose):

        train_subset_size = int(len(self.dataset) / num_folds)
        test_subset_size = int(len(self.dataset) / num_folds)
        train_fold = []
        test_fold = []
        for i in range(num_folds):
            test_fold.append((i * test_subset_size, test_subset_size))
            train_fold.append((i * train_subset_size, (i + 1) * train_subset_size))

        results = {"folds": [], "average": {}}
        classifiers = []

        pickle.dump(self, open("ML" , "wb"))
        pickle.dump(self.allLabels, open("allLabels" , "wb"))
        pickle.dump(self.dataset, open("dataset" , "wb"))
        outFiles=[]

        if self.cluster>0 :
            for i in range(num_folds):
                cwd = os.getcwd()
                stdfilename = cwd+"/train_stdout_" + str(i) + '.txt'
                print(cwd+"/train_stdout_" + str(i) + '.txt ',os.path.exists(stdfilename))
                if os.path.exists(stdfilename):
                    os.remove(stdfilename)
                errfilename = cwd+"/train_error_" + str(i) + '.txt'
                if os.path.exists(errfilename):
                    os.remove(errfilename)
                filename = cwd+"/output_" + str(i)
                if os.path.exists(filename):
                    os.remove(filename)
                outFiles.append(Path(filename))

            for i in range(num_folds):
                call("./submitTraining.sh " + str(i) + " dataset " + str(num_folds) + " ML allLabels", shell=True)

            while (np.sum([1 for f in outFiles if f.exists()]) < outFiles.__len__()):
                print(str(np.sum([1 for f in outFiles if f.exists()])))
                time.sleep(5)
        else:
            for i in range(num_folds):
                if(verbose>0):
                    print("##################################### fold " + repr(i))

                if (self.generator>0):
                    train_set = tuple(
                        (FeatureSet(self.featurize(data.text, index), index), data.label) for index, data in self.dataset.iterrows())
                else:
                    train_set = list(
                        ( (self.featurize1(data.text, index), data.label)) for index, data in self.dataset.iterrows())

                if (verbose>0):
                    print('size: ', str(train_set.__sizeof__()))
                    print('created ngram training set')

                test_data = train_set[test_fold[i][0]:test_fold[i][0] + test_fold[i][1]]
                train_data = train_set[:train_fold[i][0]] + train_set[train_fold[i][1]:]
                if (self.balancing > 0):
                    train_data = self.balance(train_data, self.balancing)

        # print('train_data: //////////////', list(dataset.values)[:train_fold[i][0]] + list(dataset.values)[train_fold[i][1]:])
        # test_data1 = train_set1[test_fold[i][0]:test_fold[i][0] + test_fold[i][1]]
        # print('test_data: //////////////',list(dataset.values)[test_fold[i][0]:test_fold[i][0]+test_fold[i][1]])
        # train_data1 = train_set1[:train_fold[i][0]] + train_set[train_fold[i][1]:]

                classifiers.append(self.train(train_data))
                if (verbose>0):
                    print('finished training')
                test_fold_classified = self.classify_many(test_data)
                if (verbose>0):
                    print('finished classify_many')

                test_fold_actual = [l for (d, l) in test_data]
                accuracy, precision, recall,f1 = scores(test_fold_actual, test_fold_classified, self.allLabels,verbose)
                results["folds"].append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1":f1})

        results["average"]["accuracy"] = np.mean(np.array([f['accuracy'] for f in results['folds']]), axis=0)
        results["average"]["precision"] = np.mean(np.array([f['precision'] for f in results['folds']]), axis=0)
        results["average"]["recall"] = np.mean(np.array([f['recall'] for f in results['folds']]), axis=0)
        results["average"]["f1"] = np.mean(np.array([f['f1'] for f in results['folds']]), axis=0)

        return results, classifiers



class FeatureSet(object):
    def __init__(self, fs,index):
        self.fs = fs
        self.index=index

    def items(self):
        return tuple(self.fs)

    def copy(self):
        return dict(self.fs).copy()
