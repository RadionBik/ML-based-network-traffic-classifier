from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class ClassifierEvaluator:
    def __init__(self, config, truth, predictions, file_suffix=''):
        self._config = config
        self.truth = truth
        self.preds = predictions
        self.classifiers = list(predictions.keys())
        self.metrics = {}
        self.conf_matrix = {}
        if file_suffix:
            self._suffix = file_suffix
        else:
            self._suffix = self._config['general']['fileSaverSuffix']
        
    def calc_scores(self):
        for classif in self.classifiers:
            calc_metrics = {
                'Accuracy': metrics.jaccard_score(self.truth, self.preds[classif], average='macro'),
                'F-score macro': metrics.f1_score(self.truth, self.preds[classif], average='macro'),
                'F-score weighted': metrics.f1_score(self.truth, self.preds[classif], average='weighted')}
            self.metrics.update({classif: calc_metrics})
        return self.metrics

    def plot_scores(self):
        scores_df = pd.DataFrame(self.calc_scores())
        print(scores_df)
        ax = scores_df.plot(kind='bar', rot=30, ylim=(0.8,1.0), grid=True).legend(bbox_to_anchor=(1.05, 1.0))
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig(self._config['report']['folderWithPlots']+os.sep
                    +'scores'+self._suffix+'.pdf')
        #plt.show()
    
    def calc_cm(self, classes):
        for classif in self.classifiers:
            cm = confusion_matrix(self.truth, self.preds[classif])
            if self._config['report'].getboolean('normalizeConfusionMatrix'):
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            df_cm = pd.DataFrame(cm, classes,classes)
            self.conf_matrix.update({classif : cm}) 
            
    def plot_cm(self, classes):

        self.calc_cm(classes)
        nrows = int(np.ceil(len(self.classifiers)/2))
        ncols = 2 if len(self.classifiers) > 1 else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[nrows*7, ncols*7])
        for index, classif in enumerate(self.classifiers):
            cm = self.conf_matrix[classif]

            if nrows==1 and ncols==1:
                axes = [axes]
            else:
                axes = axes.flatten()

            im = axes[index].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[index].set_title('CM of {} classifier'.format(classif))
            if index % 2 == 1:
                fig.colorbar(im, aspect=30, shrink=0.8, ax=axes[index])
            tick_marks = np.arange(len(classes))
            axes[index].set_xticks(tick_marks)
            axes[index].set_xticklabels(list(classes))
            plt.setp(axes[index].get_xticklabels(), rotation=45)
            axes[index].set_yticks(tick_marks)
            axes[index].set_yticklabels(list(classes))

            fmt = '.2f' if self._config['report'].getboolean('normalizeConfusionMatrix') else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[index].text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            axes[index].set_ylabel('True label')
            axes[index].set_xlabel('Predicted label')
            #fig.tight_layout()
        fig.savefig(self._config['report']['folderWithPlots']+os.sep\
                        +'Confusion_matrices'+self._suffix+'.png',dpi=400)
        plt.show()
