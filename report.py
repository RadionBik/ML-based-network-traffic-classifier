from sklearn.metrics import confusion_matrix, make_scorer
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from config_loader import Config_Init


class Classifier_Evaluator(Config_Init):
    def __init__(self, truth, predictions, suffix='', config_file='config.ini'):
        Config_Init.__init__(self, config_file)
        self.truth = truth
        self.preds = predictions
        self.classifiers = list(predictions.keys())
        self.metrics = {}
        self.conf_matrix = {}
        self._suffix = suffix
        
    def calc_scores(self):
        for classif in self.classifiers:
            calc_metrics = {'Accuracy': metrics.jaccard_similarity_score(self.truth, self.preds[classif]),
                       'F-score': metrics.f1_score(self.truth,self.preds[classif],average='weighted')}
            self.metrics.update({classif: calc_metrics})
        return self.metrics

    def plot_scores(self):
        scores_df = pd.DataFrame(self.calc_scores())
        #plt.figure(figsize=[10,6])
        ax = scores_df.plot(kind='bar', rot=45, ylim=(0.95,1), grid=True)
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig(self._config['report']['folderWithPlots']+os.sep\
                    +'scores'+self._suffix+'.pdf')
        plt.show()
    
    def calc_cm(self, classes):
        for classif in self.classifiers:
            cm = confusion_matrix(self.truth, self.preds[classif])
            if self._config['report'].getboolean('normalizeConfusionMatrix'):
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            df_cm = pd.DataFrame(cm, classes,classes)
            self.conf_matrix.update({classif : cm}) 
            
    def plot_cm(self, classes):

        self.calc_cm(classes)
        for classif in self.classifiers:
            cm = self.conf_matrix[classif]
            plt.figure(figsize=[8, 8])
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('CM of {} classifier'.format(classif))
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if self._config['report'].getboolean('normalizeConfusionMatrix') else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            #plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(self._config['report']['folderWithPlots']+os.sep\
                        +'CM_of_'+classif+self._suffix+'.pdf')
            plt.show()