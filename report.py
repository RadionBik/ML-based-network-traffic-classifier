import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, classification_report

from classifiers import ClassifierHolder
from feature_processing import Featurizer


class Reporter:
    def __init__(self, true, predicted, clf: ClassifierHolder, featurizer: Featurizer):
        self.true = true
        self.predicted = predicted
        self._featurizer = featurizer
        self._classifier = clf

    def scores(self):
        return {
            'Accuracy': jaccard_score(self.true, self.predicted, average='macro'),
            'F1 macro': f1_score(self.true, self.predicted, average='macro'),
            'F1 weighted': f1_score(self.true, self.predicted, average='weighted')
        }

    def clf_report(self):
        report = classification_report(self.true, self.predicted,
                                       target_names=self._featurizer.target_encoder.classes_,
                                       digits=3,
                                       output_dict=True)
        return pd.DataFrame(report).T

    def conf_matrix(self, normalize=None):
        return pd.DataFrame(confusion_matrix(self.true, self.predicted, normalize=normalize),
                            columns=self._featurizer.target_encoder.classes_,
                            index=self._featurizer.target_encoder.classes_)

    def plot_conf_matrix(self, normalize=None, figsize=(20, 20)):

        cm = self.conf_matrix(normalize).values
        classes = self._featurizer.target_encoder.classes_
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('CM of {} classifier'.format(self._classifier.name))
        fig.colorbar(im, aspect=30, shrink=0.8, ax=ax)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(list(classes))
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(list(classes))

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        # fig.tight_layout()
        plt.show()
