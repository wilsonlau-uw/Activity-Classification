import pandas as pd
import argparse
import csv
from sklearn.model_selection import train_test_split
from config import *

def singleton(data):
    agg = data.groupby(['label'], axis=0).agg(['count'])[GF_MODULE]
    singletions = {k:v for (k,v) in agg['count'].items() if v==1}
    return singletions

def writeData(data,iteration):
    with open('iteration'+str(iteration)+'.csv', encoding="utf-8", mode='w') as iter_output:
        writer = csv.writer(iter_output)
        writer.writerow(
            [GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION, TEXTCOL1, TEXTCOL2, TRANSLATED, BUDGET, LANGCOL,
             PRED_MODULE_TRANSLATED, PRED_INTERVENTION_TRANSLATED, CONFIDENCE_TRANSLATED,
             CORRECT_MODULE,CORRECT_INTERVENTION])
        for idx,row in data.iterrows():
            writer.writerow(row[[GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION, TEXTCOL1, TEXTCOL2, TRANSLATED, BUDGET, LANGCOL,
             PRED_MODULE_TRANSLATED, PRED_INTERVENTION_TRANSLATED, CONFIDENCE_TRANSLATED
                                 ]].fillna(''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default='3',
                        help='number of iteration')
    args = parser.parse_args()

    data = pd.read_csv('./test_output.csv',encoding='ISO-8859-1')
    data['label'] = data[GF_MODULE] + DELIMIT + data[GF_INTERVENTION]
    modules=data[GF_MODULE]
    interventions = data[GF_INTERVENTION]
    singletons= singleton(data)
    a= data[~data.label.isin(singletons)]
    X = data[~data.label.isin(singletons)]
    Y = X['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  train_size=200, test_size=200, random_state=1, stratify=Y)
    X_train=X_train.append(data[data.label.isin(singletons)])
    X_sample = data.groupby(['label',LANGCOL], group_keys=False).apply(lambda x: x.sample(max(int(len(x)*0.00001), 1)))

    writeData(X_sample,args.iteration)

