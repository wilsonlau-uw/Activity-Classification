import os.path
import warnings
import csv
import numpy as np
import pickle
import pandas as pd
from sklearn.utils import resample
from config import *
import argparse
pd.options.mode.chained_assignment = None
import sys
from os import listdir
from os.path import join



from googletrans import Translator
translator = Translator()
import ML

if os.path.exists(TRANSLATED_DATA_FILE):
    lang2EngTextMap = pickle.load(open(TRANSLATED_DATA_FILE, "rb"))
else:
    lang2EngTextMap={'fr':{},'sp':{}}

def isSameFeature(l1,l2):
    return  l1[TEXTCOL1]==l2[TEXTCOL1] and \
            l1[TEXTCOL2] == l2[TEXTCOL2] and \
            l1[GF_MODULE] == l2[GF_MODULE] and \
            l1[GF_INTERVENTION] == l2[GF_INTERVENTION] and \
            l1[MODULE] == l2[MODULE] and \
            l1[INTERVENTION] == l2[INTERVENTION]

def loadData(args,dataFile,inFile,columns=None,testData=False):
    dataDict={}
    data=[]

    with open(inFile,encoding="ISO-8859-1") as inputFile:
        reader = csv.DictReader(inputFile)

        for line in reader:
            processSuccess=True

            if (not (columns is None)):
                cols=columns
            else:
                cols = list(line.keys())
                cols.sort()
            row={}
            for c in (col for col in cols if col != TRAINFILE):

                if(c not in dataDict):
                    dataDict[c] = []

                if(testData):
                    if (float(line[CONFIDENCE_LANG])>=SEMISUPERVISED_CONFIDENCE and float(line[CONFIDENCE_TRANSLATED])>=SEMISUPERVISED_CONFIDENCE and \
                        line[PRED_MODULE_LANG]==line[PRED_MODULE_TRANSLATED] and line[PRED_INTERVENTION_LANG] ==  line[PRED_INTERVENTION_TRANSLATED]):
                        if(c == CORRECT_MODULE):
                            dataDict[c].append(line[PRED_MODULE_TRANSLATED])
                            row[c] = line[PRED_MODULE_TRANSLATED]
                        elif(c == CORRECT_INTERVENTION):
                            dataDict[c].append(line[PRED_INTERVENTION_TRANSLATED])
                            row[c] = line[PRED_INTERVENTION_TRANSLATED]
                        else:
                            dataDict[c].append(line[c])
                            row[c] = line[c]
                    else:
                        processSuccess=False
                        break
                else:
                    dataDict[c].append(line[c])
                    row[c] = line[c]

            if processSuccess:
                data.append(row)

        if args.cache >0:
            pickle.dump((dataDict,data,cols), open(dataFile, "wb"))

        dataDict[TRAINFILE]=[inFile]*len(dataDict[CORRECT_MODULE])

    return dataDict ,data,cols

def doTranslate(text):
    if(text in lang2EngTextMap):
        return lang2EngTextMap[text]
    else:
        # print(text)
        translated2 = translator.translate(text)
        lang2EngTextMap[text] = translated2.text
        return translated2.text


def outputSummary(df,data,dataDict,cols):
    with open('processed.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, cols)
        writer.writeheader()
        writer.writerows(data)

    totalItems = set(dataDict[CORRECT_MODULE])
    print('total items : ', totalItems.__len__())
    ptable = pd.pivot_table(df, values=TEXTCOL2, index=[ CORRECT_MODULE, CORRECT_INTERVENTION, LANGCOL],
                            aggfunc=np.count_nonzero, fill_value=0)

    writer = pd.ExcelWriter('output_categories.xlsx')
    ptable.to_excel(writer)
    writer.save()


def pickleExt(args):
    return '_c'+ args.classifier+ '_d'+str(args.degree)+ '_w'+str(args.window)+'_b'+str(args.balance)+'_g'+str(args.generator)+ '.pickle'

def prepareforML(df,disease,lang,translate):
    sanityCheck={}
    if(translate):
        df = df[df[LANGCOL].str.startswith(disease)]
        mlData = df[[  TEXTCOL1, TEXTCOL2, GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION,LANGCOL,BUDGET,TRAINFILE]]
        mlData['label'] = df[CORRECT_MODULE] + DELIMIT + df[CORRECT_INTERVENTION]
        texts=[]
        raws=[]
        for i,row in mlData.iterrows():

            raws.append('***' + row[GF_MODULE].replace(' ','_') + ' ' +
                         '***' + row[GF_INTERVENTION].replace(' ','_') + ' ' +
                         '***' + row[MODULE].replace(' ','_') + ' ' +
                         '***' + row[INTERVENTION].replace(' ','_') + ' ' +
                         '***' + (row[TEXTCOL1].split()[0] if len(row[TEXTCOL1])>0 else '') + ' ' + row[TEXTCOL2].lower())

            if(not row['disease_lang_concat'].endswith('eng')):
                translatedTxt=doTranslate(row[TEXTCOL2])

                texts.append( '***'+ row[GF_MODULE].replace(' ','_')  + ' ' +
                              '***' + row[GF_INTERVENTION].replace(' ','_') + ' ' +
                              '***' + row[MODULE].replace(' ','_') + ' ' +
                              '***' + row[INTERVENTION].replace(' ','_') + ' ' +
                              '***' + (row[TEXTCOL1].split()[0] if len(row[TEXTCOL1])>0 else '')  + ' ' + translatedTxt.lower() )
            else:
                texts.append( '***'+ row[GF_MODULE].replace(' ','_')  + ' ' +
                              '***' + row[GF_INTERVENTION].replace(' ','_') + ' ' +
                              '***' + row[MODULE].replace(' ','_') + ' ' +
                              '***' + row[INTERVENTION].replace(' ','_') + ' ' +
                              '***' +  (row[TEXTCOL1].split()[0] if len(row[TEXTCOL1])>0 else '') + ' ' + row[TEXTCOL2].lower() )

        mlData['text']=texts
        mlData['raw']=raws
        pickle.dump(lang2EngTextMap, open(TRANSLATED_DATA_FILE, "wb"))

    else:
        df = df[df['disease_lang_concat'].str.endswith(disease+lang)]
        mlData = df[[ TEXTCOL1, TEXTCOL2, GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION,LANGCOL,BUDGET,TRAINFILE]]
        mlData['label'] = df[CORRECT_MODULE] + DELIMIT + df[CORRECT_INTERVENTION]
        mlData['text'] =  [ '***'+ v[GF_MODULE].replace(' ','_')  + ' ' +
                              '***' + v[GF_INTERVENTION].replace(' ','_') + ' ' +
                              '***' + v[MODULE].replace(' ','_') + ' ' +
                              '***' + v[INTERVENTION].replace(' ','_') + ' ' +
                              '***' + (v[TEXTCOL1].split()[0] if len(v[TEXTCOL1])>0 else '')+ ' ' for i,v in mlData.iterrows()]  + mlData[TEXTCOL2]

    mlData = mlData.sample(n=len(mlData), random_state=3)


    # check for inconsistent labels and remove the ones that are coming from the test results
    for i,d in mlData.iterrows():
        if(d['text'] not in sanityCheck ):
            s=set()
            s.add(d['label'])
            sanityCheck[d['text']] = s
        else:
            sanityCheck[d['text']].add(d['label'])

    print('*'*20+' Checking for inconsistent labeling ' + '*'*20)
    failedCheck = [s for s in sanityCheck if len(sanityCheck[s])>1]
    for s in failedCheck:
        msg=''
        msg+='='*10+'\n'
        formatted = s.split('***')
        msg+=GF_MODULE+' : '+formatted[1]+'\n'
        msg+=GF_INTERVENTION + ' : ' + formatted[2]+'\n'
        msg+=MODULE + ' : ' + formatted[3]+'\n'
        msg+=INTERVENTION + ' : ' + formatted[4]+'\n'
        msg+=TEXTCOL1 + ' '+TEXTCOL2+' : ' + formatted[5]+'\n'
        rows = mlData.loc[mlData['text'] == s]
        mlData = mlData[~ (mlData.text.isin([s]) & mlData.file.str.endswith('test_output.csv'))]
        rows = mlData.loc[mlData['text'] == s]
        if(len(set(rows[TRAINFILE]))==1):
            continue
        msg+='-->\n' + str('\n'.join(set([CORRECT_MODULE + ' : ' + r['label'].split(DELIMIT)[
            0] + '   ' + CORRECT_INTERVENTION + ' : ' + r['label'].split(DELIMIT)[1] + '   (' + r[TRAINFILE] + ')' for
                                           x, r in rows.iterrows()])))+'\n'
        msg+='=' * 10+'\n'
        print(msg)

    return mlData,df


def prepareTraining(df,disease,lang,translate):
    mlData,df_train = prepareforML(df, disease,lang,translate)
    return mlData

def createNGram(args,mlData,disease,lang):
    if (args.cache > 0):
        dataFile = 'nGramModel'+'_'+disease+lang + pickleExt(args)
        if not os.path.exists(dataFile):
            nGramModel = ML.Ngram(args.classifier, mlData, args.generator, args.degree, args.remove, args.balance,
                                  args.cluster)
            pickle.dump(nGramModel, open(dataFile, "wb"))

        nGramModel = pickle.load(open(dataFile, "rb"))
    else:
        nGramModel = ML.Ngram(args.classifier, mlData, args.generator, args.degree, args.remove, args.balance, args.cluster)

    return nGramModel

def trainModel(df, disease, lang,args):
    mlData=prepareTraining(df,disease,lang,args.translate>0)
    nGramModel = createNGram(args, mlData, disease, lang)

    print('number of ngrams in '+disease+' '+lang+':', str(len(nGramModel.getAllNgrams())))

    train_set = list(
        ((nGramModel.featurize1(data.text, index), data.label)) for index, data in nGramModel.dataset.iterrows())

    if (args.balance > 0):
        train_data = nGramModel.balance(train_set, args.balance )
    else:
        train_data = train_set

    nGramModel.train(train_data)
    return nGramModel


def testModel(trainModel,df,disease,lang,args):
    mlData,df_test = prepareforML(df, disease,lang,args.translate>0)
    test_set = list(
        ((trainModel.featurize2(data.text), data.label)) for index, data in mlData.iterrows())

    test_classified = trainModel.classify_many(test_set)
    return test_classified,mlData

def testModels(trainModels,trainModels_translated, df,disease):
    mlData,df_test = prepareforML(df, disease,'',translate=True)
    test_set = list(
        ((trainModels_translated.featurize2(data.text),
          trainModels[data.disease_lang_concat[len(disease):]].featurize2(data.raw),
          data.label, data.disease_lang_concat)) for index, data in mlData.iterrows())

    test_classified ={'translate':[],'lang':[]}
    for (d1,d2,l, d3) in test_set:
        test_classified['translate'].append( trainModels_translated.classify_predict(d1))
        test_classified['lang'].append( trainModels[d3[len(disease):]].classify_predict(d2))

    return test_classified,mlData

def createTestParams(classifier,degree,balance,remove,translate):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', default=classifier)
    parser.add_argument('--degree', default=degree)
    parser.add_argument("--window", default=0, type=int,
                        help='sentence window (default: 0, all ngrams)')
    parser.add_argument("--balance", default=balance)
    parser.add_argument("--generator", default=0, type=int,
                        help='using generator function (default: 0)')
    parser.add_argument("--cluster", default=0, type=int,
                        help='using clusters (default: 1)')
    parser.add_argument("--remove", default=remove, type=int )
    parser.add_argument("--cache", default=0, type=int,
                        help='save and load cached data [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--cv", default=1, type=int,
                        help='cross validate [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--verbose", default=0, type=int,
                        help='debug message [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--translate", default=translate, type=int )

    return parser.parse_args()

def runAllTests(df_train, translate):
    all_results={}
    argsList=[
        createTestParams(classifier='P', degree=[1,2,3,4], balance=0, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=0, remove=0, translate=translate),
        createTestParams(classifier='S', degree=[1], balance=0, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=1, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=0, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=3, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=4, remove=0, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=3, remove=1, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=4, remove=1, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=3, remove=3, translate=translate),
        createTestParams(classifier='P', degree=[1], balance=4, remove=3, translate=translate),
        createTestParams(classifier='P-Select', degree=[1], balance=3, remove=1, translate=translate),
        createTestParams(classifier='P-Select', degree=[1], balance=4, remove=1, translate=translate),
        createTestParams(classifier='P-KBest', degree=[1], balance=3, remove=1, translate=translate),
        createTestParams(classifier='P-KBest', degree=[1], balance=4, remove=1, translate=translate),
        createTestParams(classifier='RF', degree=[1], balance=4, remove=1, translate=translate),

    ]
    for args in argsList:

        print(args)

        mlData = prepareTraining(df_train, 'malaria', 'fr', args.translate > 0)
        allLabels = sorted(list(set(mlData['label'])))
        nGramModel_train = createNGram(args, mlData, 'malaria', 'fr')

        warnings.filterwarnings('ignore')
        results, classifiers = nGramModel_train.cross_validate(10, args.verbose)
        results["features"] = len(nGramModel_train.all_ngrams)

        # print("##################################### Summary ##############################################")
        label_result = {}
        for i,label in enumerate(allLabels):

            label_result[label]={'accuracy': results["average"]["accuracy"][i],
            'precision': results["average"]["precision"][i],
            'recall': results["average"]["recall"][i],
            'f1': results["average"]["f1"][i]}
            # print(allLabels[i])
            # print("Accuracy: {:10.2f}".format(results["average"]["accuracy"][i]))
            # print("Precision: {:10.2f}".format(results["average"]["precision"][i]))
            # print("Recall: {:10.2f}".format(results["average"]["recall"][i]))
            # print("F1: {:10.2f}".format(results["average"]["f1"][i]))
            #
            # print('')

        all_results[str(args)]=label_result
        sys.stdout.flush()

    accuracies={}
    precisions={}
    recalls={}
    f1s={}
    for args in argsList:
        print(args)
        res=all_results[str(args)]
        for i, label in enumerate(allLabels):
            if(label not in accuracies):
                accuracies[label]=[]
                precisions[label]=[]
                recalls[label]=[]
                f1s[label]=[]

            accuracies[label].append(res[label]["accuracy"])
            precisions[label].append(res[label]["precision"] )
            recalls[label].append(res[label]["recall"] )
            f1s[label].append(res[label]["f1"] )


    with open('testresults_alltests.csv' if translate ==0 else 'testresults_alltests_translated.csv', 'w') as test_results:
        writer = csv.writer(test_results)
        writer.writerow(['test case']+ [str(a) for a in argsList])
        for i, label in enumerate(allLabels):
            writer.writerow(['']+[label]*len(argsList))
            writer.writerow(['precision:']+[a for a in precisions[label]  ])
            writer.writerow(['recall:']+[a for a in recalls[label]  ])
            writer.writerow(['f1:']+[a for a in f1s[label]  ])
            writer.writerow('')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier',default='P-Select',
                        help='classifier (N/S/LS/L/M/P/P-Select/P-KBest/RF)')
    parser.add_argument('--degree',default=[1],type=int,
                        help='list of degree of ngram (list)')
    parser.add_argument("--window", default=0,type=int,
                        help='sentence window (default: 0, all ngrams)')
    parser.add_argument("--balance", default=3, type=int,
                        help='balance with resample [0: None | 1: Upsample | 2: downSample | 3: RandomOverSample | 4: SMOTE] (default: 0)')
    parser.add_argument("--generator", default=0, type=int,
                        help='using generator function (default: 0)')
    parser.add_argument("--cluster", default=0, type=int,
                        help='using clusters (default: 1)')
    parser.add_argument("--remove", default=1, type=int,
                        help='remove ngram with frequency <= this threshold (default: 0)')
    parser.add_argument("--cache", default=0, type=int,
                        help='save and load cached data [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--use", default=2, type=int,
                        help='cross validate/single test/full run [0: C.V. | 1: Single Test | 2: Full] (default: 0)')
    parser.add_argument("--verbose", default=0, type=int,
                        help='debug message [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--translate", default=1, type=int,
                        help='translate to English [1: Yes | 0: No] (default: 0)')
    parser.add_argument("--iterations", default='all',
                        help='list of validated files from iterations (default: all)')
    parser.add_argument("--semisupervised", default='iterations/iteration3/test_output.csv',
                        help='self-training using past predicted data (default: None)')

    args = parser.parse_args()
    print(args)
    dataFile= 'data'+ pickleExt(args)
    trainingFile = 'nlp_training_sample.csv'
    if(args.cache>0):
        if not os.path.exists(dataFile):
            dataDict, data, cols = loadData(args,dataFile,trainingFile)

        (dataDict, data, cols) = pickle.load(open(dataFile, "rb"))
    else:
        dataDict, data, cols = loadData(args, dataFile,trainingFile)


    if(not (args.iterations is None)):
        iterationFiles = []
        if(args.iterations =='all'):
            iterFolders = [join('./iterations', f) for f in listdir('./iterations') if f.startswith('iteration')]
            for f in iterFolders:
                iterationFiles.extend([join(f, f1) for f1 in listdir(f) if f1.startswith('iteration') and f1.endswith('.csv')])

        elif len(args.iterations)>0:
            iterationFiles = args.iterations

        for file in iterationFiles:
            validatedDataDict, validatedData, validatedCols = loadData(args, dataFile, file, columns=dataDict.keys())
            data.extend(validatedData)
            for c in dataDict.keys():
                dataDict[c].extend(validatedDataDict[c])


    if (args.semisupervised is not None):
        testDataDict, testData, testCols = loadData(args, dataFile, args.semisupervised, columns=dataDict.keys(),testData=True)
        data.extend(testData)
        for c in dataDict.keys():
            dataDict[c].extend(testDataDict[c])

    df_train = pd.DataFrame.from_dict(dataDict)

    # outputSummary(df_train,data,dataDict,cols)


    if (args.use == 0):  # CV all tests

        runAllTests(df_train,0)
        runAllTests(df_train, 1)
        pickle.dump(lang2EngTextMap, open(TRANSLATED_DATA_FILE, "wb"))

    elif (args.use == 1):
        testFile = 'nlp_test_sample.csv'
        testdataDict, testdata, testcols = loadData(args, dataFile, testFile)
        df_test = pd.DataFrame.from_dict(testdataDict)
        args.translate=0
        nGramModel_malaria_translated = trainModel(df_train, 'malaria', 'fr', args)

        test_classified, df_test_model = testModel(nGramModel_malaria_translated, df_test, 'malaria', 'fr', args)
        pickle.dump(lang2EngTextMap, open(TRANSLATED_DATA_FILE, "wb"))
        # output results
        with open('test_ouput_lang.csv', encoding="ISO-8859-1", mode='w') as test_output:
            writer = csv.writer(test_output)
            writer.writerow(
                [GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION, TEXTCOL1, TEXTCOL2, BUDGET, LANGCOL, PRED_MODULE_LANG,
                 PRED_INTERVENTION_LANG, CONFIDENCE_LANG])
            j = 0
            for i, item in df_test_model.iterrows():
                row = [item[GF_MODULE], item[GF_INTERVENTION], item[MODULE], item[INTERVENTION], item[TEXTCOL1],
                       item[TEXTCOL2], item[BUDGET], item[LANGCOL] \
                    , test_classified[j].split(DELIMIT)[0], test_classified[j].split(DELIMIT)[1],
                       nGramModel_malaria_translated.predictProbs[j].max()]
                writer.writerow(row)
                j = j + 1

    else:
        testFile = 'nlp_test_sample.csv'  # 73386 records
        testdataDict, testdata, testcols = loadData(args, dataFile, testFile)
        df_test = pd.DataFrame.from_dict(testdataDict)
        DISEASES = ['malaria','hiv','tb']
        LANGS=['eng','fr','esp']
        args.translate=1
        nGramModels_translated=dict.fromkeys(DISEASES)

        for disease in DISEASES:
            nGramModels_translated[disease]=trainModel(df_train, disease, '',args)

        args.translate = 0
        nGramModels=dict(zip(dict.fromkeys(DISEASES),[dict.fromkeys(LANGS),dict.fromkeys(LANGS),dict.fromkeys(LANGS)] ))
        df_test_models = dict(zip(dict.fromkeys(DISEASES),[dict.fromkeys(LANGS),dict.fromkeys(LANGS) ,dict.fromkeys(LANGS)] ))

        tests_classified=dict.fromkeys(DISEASES)

        for disease in DISEASES:
            for lang in LANGS:
                disease_lang = disease + lang
                if(disease_lang in [d for d in df_train[LANGCOL]]):
                    nGramModels[disease][lang]=trainModel(df_train,disease,lang,args)

        for disease in DISEASES:
            tests_classified[disease], df_test_models[disease]= testModels(nGramModels[disease],
                                                                                            nGramModels_translated[disease],
                                                                                            df_test[df_test.disease_lang_concat.str.startswith(disease)],
                                                                                            disease)
        pickle.dump((tests_classified,df_test_models), open('tmp.pickle', "wb"))
        (tests_classified, df_test_models) = pickle.load(open('tmp.pickle','rb'))
        # output results
        with open('test_output.csv',encoding="utf-8", mode='w') as test_output:
            writer = csv.writer(test_output)
            writer.writerow([ GF_MODULE,GF_INTERVENTION,MODULE,INTERVENTION,TEXTCOL1,TEXTCOL2,TRANSLATED,BUDGET,LANGCOL,
                              PRED_MODULE_TRANSLATED,PRED_INTERVENTION_TRANSLATED,CONFIDENCE_TRANSLATED,
                              PRED_2_MODULE_TRANSLATED, PRED_2_INTERVENTION_TRANSLATED, CONFIDENCE_2_TRANSLATED,
                              PRED_3_MODULE_TRANSLATED, PRED_3_INTERVENTION_TRANSLATED, CONFIDENCE_3_TRANSLATED,
                              PRED_MODULE_LANG,PRED_INTERVENTION_LANG,CONFIDENCE_LANG,
                              PRED_2_MODULE_LANG, PRED_2_INTERVENTION_LANG, CONFIDENCE_2_LANG,
                              PRED_3_MODULE_LANG, PRED_3_INTERVENTION_LANG, CONFIDENCE_3_LANG
                              ])

            for disease in DISEASES:
                j = 0
                for i, item in df_test_models[disease].iterrows():
                    row = [item[GF_MODULE],	item[GF_INTERVENTION], item[MODULE], item[INTERVENTION], item[TEXTCOL1],item[TEXTCOL2], doTranslate(item[TEXTCOL2]) if not item[LANGCOL].endswith('eng') else '', item[BUDGET],item[LANGCOL] \
                          , tests_classified[disease]['translate'][j][0][0].split(DELIMIT)[0],
                           tests_classified[disease]['translate'][j][0][0].split(DELIMIT)[1],
                           tests_classified[disease]['translate'][j][0][1]
                        , tests_classified[disease]['translate'][j][1][0].split(DELIMIT)[0],
                           tests_classified[disease]['translate'][j][1][0].split(DELIMIT)[1],
                           tests_classified[disease]['translate'][j][1][1]
                        , tests_classified[disease]['translate'][j][2][0].split(DELIMIT)[0],
                           tests_classified[disease]['translate'][j][2][0].split(DELIMIT)[1],
                           tests_classified[disease]['translate'][j][2][1]
                    , tests_classified[disease]['lang'][j][0][0].split(DELIMIT)[0],
                          tests_classified[disease]['lang'][j][0][0].split(DELIMIT)[1],
                          tests_classified[disease]['lang'][j][0][1]
                        , tests_classified[disease]['lang'][j][1][0].split(DELIMIT)[0],
                           tests_classified[disease]['lang'][j][1][0].split(DELIMIT)[1],
                           tests_classified[disease]['lang'][j][1][1]
                        , tests_classified[disease]['lang'][j][2][0].split(DELIMIT)[0],
                           tests_classified[disease]['lang'][j][2][0].split(DELIMIT)[1],
                           tests_classified[disease]['lang'][j][2][1]
                    ]
                    writer.writerow(row)
                    j=j+1

        pickle.dump(lang2EngTextMap, open(TRANSLATED_DATA_FILE, "wb"))