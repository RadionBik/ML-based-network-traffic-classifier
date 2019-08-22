import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import os
import pandas as pd


class TransformNotFound(FileNotFoundError):
    def __init(self, filename):
        super().__init__('Transform {} was not found, please check it exists or run training first'.format(
            filename
        ))


class FeatureTransformer:
    """
    fit_transform() processes raw targets and features pandas objects,
    learning new labels and scalers and one-hot encoding selected
    ones.

    IMPORTANT: USE load_transform() if you want to use
    TRAINED classifiers

    returns X_train, y_train, X_test, y_test
    """

    def __init__(self, config, categ_features=None, file_suffix=None, feature_flags=None):

        self._config = config
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.one_hot = OneHotEncoder()
    
        if file_suffix:
            self._suffix = file_suffix
        else:
            self._suffix = self._config['general']['fileSaverSuffix']

        self.random_seed = int(self._config['offline']['randomSeed'])

        self._folder_pref = self._config['general']['classifiers_folder']+os.sep
        self._one_hot_file = self._folder_pref+'one_hot'+self._suffix+'.dat'
        self._scaler_file = self._folder_pref+'scaler'+self._suffix+'.dat'
        self._le_file = self._folder_pref+'le'+self._suffix+'.dat'
        if not categ_features:
            self.categ_features = ['client_found_tcp_flags', 'server_found_tcp_flags']
        else:
            self.categ_features = categ_features
        self._split = float(self._config['offline']['splitRatio'])
        if feature_flags:
            self.consider_iat = feature_flags[0]
            self.consider_tcp_flags = feature_flags[1]
        else:
            self.consider_iat = self._config['parser'].getboolean('considerIAT')
            self.consider_tcp_flags = self._config['parser'].getboolean('considerTCPflags')

    def _fit_transform_scale_and_labels(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_labeled = self.le.fit_transform(y)
        joblib.dump(self.scaler, self._scaler_file)
        joblib.dump(self.le, self._le_file)
        return X_scaled, y_labeled

    def _load_transform_scale_and_labels(self, X, y):
        self.scaler = joblib.load(self._scaler_file)
        self.le = joblib.load(self._le_file)

        X_scaled = self.scaler.transform(X)
        y_labeled = self.le.transform(y)

        return X_scaled, y_labeled

    def _transform_scale_and_labels(self, X, y):
        return self.scaler.transform(X), self.le.transform(y)

    def _fit_transform_one_hot(self, features) -> tuple:
        selected = features[self.categ_features]
        one_hot = self.one_hot.fit_transform(selected).toarray()
        joblib.dump(self.one_hot, self._one_hot_file)
        return one_hot, features.drop(self.categ_features, axis=1)

    def _load_transform_one_hot(self, features) -> tuple:
        try:
            self.one_hot = joblib.load(self._one_hot_file)
        except FileNotFoundError as exc:
            raise TransformNotFound(self._one_hot_file) from exc
        selected = features[self.categ_features]
        one_hot = self.one_hot.transform(selected).toarray()
        return one_hot, features.drop(self.categ_features, axis=1)

    def _split_and_label_features(self, one_hot_features, features, targets):
        if not self.consider_iat:
            features = features.drop(list(features.filter(regex='iat')), axis=1)

        F_tr, F_test, X_oh_tr, X_oh_test, t_tr, t_test = train_test_split(features,
                                                                          one_hot_features,
                                                                          targets,
                                                                          shuffle=True,
                                                                          test_size=self._split,
                                                                          stratify=targets,
                                                                          random_state=self.random_seed)

        X_tr, y_tr = self._fit_transform_scale_and_labels(F_tr, t_tr)
        X_test, y_test = self._transform_scale_and_labels(F_test, t_test)

        if self.consider_tcp_flags:
            return np.hstack([X_tr, X_oh_tr]), y_tr, np.hstack([X_test, X_oh_test]), y_test
        else:
            return X_tr, y_tr, X_test, y_test

    def fit_transform(self, features, targets):
        one_hot_features, features = self._fit_transform_one_hot(features)
        return self._split_and_label_features(one_hot_features, features, targets)

    def load_transform(self, features, targets):
        one_hot_features, features = self._load_transform_one_hot(features)
        return self._split_and_label_features(one_hot_features, features, targets)


def _rename_protocols_inplace(flow_features):
    #  TODO: make a non-modifying version
    flow_features.replace('SSL_No_Cert', 'SSL', inplace=True)
    flow_features.replace('Unencrypted_Jabber', 'Jabber', inplace=True)
    flow_features.replace('Viber', 'DNS', inplace=True)
    return flow_features


def _filter_apps(flow_features, minflows):
    found_apps = flow_features.proto.value_counts()
    print('found apps:', found_apps)
    good_apps = [app_index for app_index, flow_count in
                 zip(found_apps.index, found_apps)
                 if flow_count >= minflows]

    flow_features = flow_features[flow_features['proto'].map(lambda x: x in good_apps)]
    return flow_features


def read_csv(filename):
    """ a simple wrapper for pandas """
    return pd.read_csv(filename,
                       sep='|',
                       index_col=0)


def prepare_data(data, min_flows_per_app: int = 20):
    """
    prepare_data() removes rare protocols and flows, splits DataFrame
    into target vector and feature matrix
    """
    data = _rename_protocols_inplace(data)
    data = _filter_apps(data, min_flows_per_app)

    data.drop('subproto', axis=1, inplace=True)
    data.fillna(0, inplace=True)
    features, protocols = data.drop('proto', axis=1), data['proto']
    return features, protocols
