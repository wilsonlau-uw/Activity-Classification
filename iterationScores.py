import pandas as pd
import numpy as np
import argparse
import csv
from imblearn.metrics import sensitivity_specificity_support
from config import *
from os import listdir
from os.path import join

def singleton(data):
    agg = data.groupby(['label'], axis=0).agg(['count'])[GF_MODULE]
    singletions = {k:v for (k,v) in agg['count'].items() if v==1}
    return singletions

def writeData(data,iteration):
    with open('iteration'+str(iteration)+'.csv', encoding="utf-8", mode='w') as iter_output:
        writer = csv.writer(iter_output)
        writer.writerow(
            [GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION, TEXTCOL1, TEXTCOL2, TRANSLATED, BUDGET, LANGCOL,
             PRED_MODULE_TRANSLATED, PRED_INTERVENTION_TRANSLATED, CONFIDENCE_TRANSLATED, PRED_MODULE_LANG,
             PRED_INTERVENTION_LANG, CONFIDENCE_LANG,CORRECT_MODULE,CORRECT_INTERVENTION])
        for idx,row in data.iterrows():
            writer.writerow(row[[GF_MODULE, GF_INTERVENTION, MODULE, INTERVENTION, TEXTCOL1, TEXTCOL2, TRANSLATED, BUDGET, LANGCOL,
             PRED_MODULE_TRANSLATED, PRED_INTERVENTION_TRANSLATED, CONFIDENCE_TRANSLATED, PRED_MODULE_LANG,
             PRED_INTERVENTION_LANG, CONFIDENCE_LANG]])

def plot(disease, data):
    subSet = data[data.disease_lang_concat.str.startswith(disease)]
    labels = list(set(subSet['true_label']))
    labels.sort()
    sensitivity, specificity, support = sensitivity_specificity_support(subSet['true_label'],
                                                                        subSet['predicted_translated'],
                                                                        labels=labels, average=None)
    df=pd.DataFrame({'Factor': labels,
                      'Sensitivity': sensitivity,
                      'Specificity': specificity})
    tidy= (df.set_index('Factor').stack().reset_index().rename(columns={'level_1':'Variable',0:'Value'}))

    sns.set(font_scale=0.8)

    # plt.interactive(False)
    a4_dims = (15, 8)
    width=0.35
    fig, ax = plt.subplots(figsize=a4_dims)
    y_pos=np.arange(len(labels))
    sen = ax.barh( y_pos, sensitivity,  width,  color='b')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    spe = ax.barh(  y_pos+width,specificity,  width,  color='y')
    ax.set_xlabel('scores')
    ax.set_title(disease)
    ax.legend((sen,spe),('Sensitivity','Specificity'))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode='expand', borderaxespad=0.)
    plt.tight_layout(h_pad=1)
    plt.show()

    plt.savefig(disease+'.png')


    # sns.barplot(ax=ax, data=tidy,  palette="BuGn_d", orient='h',y='Factor',x='Value')
    # sns.despine(fig)
    # # ax.legend(ncol=2, loc="upper left", frameon=True)
    # mng = plt.get_current_fig_manager()
    # # mng.window.showMaximized()
    # plt.title(disease, fontdict={'fontsize': 10})
    # plt.legend(bbox_to_anchor=(0.,1.02,1.,.102), loc=3, ncol=2, mode='expand',borderaxespad=0.)
    # plt.tight_layout(h_pad=1)

    # sns.barplot(ax=ax, data=pd.DataFrame.from_records([specificity], columns=labels), orient='h', palette="RdBu_r")
    #
    # plt.title(disease + ' specificity', fontdict={'fontsize': 10})
    # plt.tight_layout(h_pad=1)

    int=1

def scoreIteration(args):
    data = pd.read_csv(args.filepath, encoding='ISO-8859-1')
    data['predicted_translated'] = data[GF_MODULE] + DELIMIT + data[GF_INTERVENTION]
    data['true_label'] = data[CORRECT_MODULE] + DELIMIT + data[GF_INTERVENTION]
    labels = list(set(data['true_label']))


    sensitivity, specificity, support = sensitivity_specificity_support(data['true_label'],
                                                                        data['predicted_translated'],
                                                                     labels=labels, average='macro')
    print(args.filepath)
    print('sensitivity - '+ str(sensitivity))
    print('specificity - '+str(specificity))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default='all',
                        help='validated iteration file')
    parser.add_argument('--charts', default = 0, type = int,
                        help='plotting charts')

    args = parser.parse_args()

    if(args.filepath == 'all'):
        iterFolders = [join('./iterations', f) for f in listdir('./iterations') if f.startswith('iteration')]
        iterationFiles=[]
        for f in iterFolders:
            iterationFiles.extend( [join(f, f1)for f1 in listdir(f)  if f1.startswith('iteration') and f1.endswith('.csv')])

        for f in iterationFiles:
            args.filepath = f
            scoreIteration(args)
    else:
        scoreIteration(args)



    if(args.charts>0):
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns;

        matplotlib.interactive(True)
        data = pd.read_csv(args.filepath,encoding='ISO-8859-1')
        data['predicted_translated'] = data[GF_MODULE] + DELIMIT + data[GF_INTERVENTION]
        data['true_label'] = data[CORRECT_MODULE] + DELIMIT + data[GF_INTERVENTION]
        labels=list(set(data['true_label']))
        plot('malaria',data)
        plot('hiv', data)
        plot('tb', data)

        sensitivity,specificity,support= sensitivity_specificity_support(data['true_label'], data['predicted_translated'],
                                                                         labels=labels, average='macro')
        print('sensitivity - '+ str(sensitivity))
        print('specificity - '+str(specificity))
        print(support)
        uniform_data = np.random.rand(10, 12)
        iris = sns.load_dataset("iris")
        sns.set(font_scale=0.5)

        # plt.interactive(False)
        a4_dims = (20,8)
        fig, ax = plt.subplots(figsize=a4_dims)
        ax = sns.barplot(ax=ax,data=pd.DataFrame.from_records([sensitivity], columns=labels), orient='h', palette="BuGn_d")
        mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        plt.tight_layout(h_pad=1)
        # plt.show()
        # plt.yticks(15 * np.arange(len(labels)))
        # plt.setp(ax.set_yticklabels(labels))

        # f, (ax1, ax2, ax3) = plt.subplots(3, 1,  sharex=True)
        #
        # sns.barplot(labels, sensitivity, palette="BuGn_d",c ax=ax1)
        # ax1.set_ylabel("sensitivity")
        # sns.barplot(labels, specificity, palette="RdBu_r", ax=ax2)
        # ax2.set_ylabel("specificity")
        # sns.barplot(labels, support, palette="Set3", ax=ax3)
        # ax3.set_ylabel("support")
        #
        # sns.despine(bottom=True)
        # plt.setp(f.axes, yticks=[])
        plt.tight_layout(h_pad=1)