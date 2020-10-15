from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_classifiers.knn_cosine import KNeighborsCosineClassifier, KNeighborsLshClassifier

REGISTERED_CLASSES = {
    cls.__name__: cls for cls in [
        MLPClassifier,
        LinearSVC,
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
        LogisticRegression,
        OneVsOneClassifier,
        KNeighborsClassifier,
        KNeighborsCosineClassifier,
        KNeighborsLshClassifier
    ]
}
