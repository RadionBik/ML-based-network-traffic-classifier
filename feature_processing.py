import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


class Feature_Extractor:
    def __init__(self, consider_iat=True):
        self._consider_iat = consider_iat

    def extract_features(self, raw_df, consider_iat=True):
        # print(raw_df.head())
        stats = {}

        client_bulks = raw_df[(raw_df['transp_payload'] > 0) &
                              (raw_df['is_client'] == 1)
                              ]['transp_payload']

        server_bulks = raw_df[(raw_df['transp_payload'] > 0) &
                              (raw_df['is_client'] == 0)
                              ]['transp_payload']

        client_packets = raw_df[raw_df['is_client'] == 0
                                ]['ip_payload']

        server_packets = raw_df[raw_df['is_client'] == 0
                                ]['ip_payload']

        fault_avoider = (
            lambda values, index=0: values.iloc[index] if len(values) > index else 0)

        stats.update({
            'proto': raw_df['proto'].iloc[0],
            'subproto': raw_df['subproto'].iloc[0],
            'is_tcp': raw_df['is_tcp'].iloc[0],

            'client_found_tcp_flags': sorted(list(set(raw_df[raw_df['is_client'] == 1]['tcp_flags']))),
            'server_found_tcp_flags': sorted(list(set(raw_df[raw_df['is_client'] == 0]['tcp_flags']))),

            'client_tcp_window_mean': raw_df[raw_df['is_client'] == 1]['tcp_win'].mean(),
            'server_tcp_window_mean': raw_df[raw_df['is_client'] == 0]['tcp_win'].mean(),

            'client_bulk0': fault_avoider(client_bulks, 0),
            'client_bulk1': fault_avoider(client_bulks, 1),
            'server_bulk0': fault_avoider(server_bulks, 0),
            'server_bulk1': fault_avoider(server_bulks, 1),

            'client_packet0': fault_avoider(client_packets, 0),
            'client_packet1': fault_avoider(client_packets, 1),
            'server_packet0': fault_avoider(server_packets, 0),
            'server_packet1': fault_avoider(server_packets, 1),

            'server_bulk_max': server_bulks.max(),
            'server_bulk_min': server_bulks.min(),
            'server_bulk_mean': server_bulks.mean(),
            'server_bulk_median': server_bulks.quantile(.5),
            'server_bulk_25q': server_bulks.quantile(.25),
            'server_bulk_75q': server_bulks.quantile(.75),
            'server_bulks_bytes': server_bulks.sum(),
            'server_bulks_number': len(server_bulks),

            'client_bulk_max': client_bulks.max(),
            'client_bulk_min': client_bulks.min(),
            'client_bulk_mean': client_bulks.mean(),
            'client_bulk_median': client_bulks.quantile(.5),
            'client_bulk_25q': client_bulks.quantile(.25),
            'client_bulk_75q': client_bulks.quantile(.75),
            'client_bulks_bytes': client_bulks.sum(),
            'client_bulks_number': len(client_bulks),

            'server_packet_max': server_packets.max(),
            'server_packet_min': server_packets.min(),
            'server_packet_mean': server_packets.mean(),
            'server_packet_median': server_packets.quantile(.5),
            'server_packet_25q': server_packets.quantile(.25),
            'server_packet_75q': server_packets.quantile(.75),
            'server_packets_bytes': server_packets.sum(),
            'server_packets_number': len(server_packets),

            'client_packet_max': client_packets.max(),
            'client_packet_min': client_packets.min(),
            'client_packet_mean': client_packets.mean(),
            'client_packet_median': client_packets.quantile(.5),
            'client_packet_25q': client_packets.quantile(.25),
            'client_packet_75q': client_packets.quantile(.75),
            'client_packets_bytes': client_packets.sum(),
            'client_packets_number': len(client_packets),
        })

        if consider_iat:
            iat_client = pd.to_timedelta(pd.Series(
                raw_df[raw_df['is_client'] == 1].index).diff().fillna('0')) / pd.offsets.Second(1)
            iat_client.index = raw_df[raw_df['is_client'] == 1].index

            iat_server = pd.to_timedelta(pd.Series(
                raw_df[raw_df['is_client'] == 0].index).diff().fillna('0')) / pd.offsets.Second(1)
            iat_server.index = raw_df[raw_df['is_client'] == 0].index

            raw_df['IAT'] = pd.concat([iat_server, iat_client])

            client_iats = raw_df[raw_df['is_client'] == 1
                                 ]['IAT']
            server_iats = raw_df[raw_df['is_client'] == 0
                                 ]['IAT']

            stats.update({
                'client_iat_mean': client_iats.mean(),
                'client_iat_median': client_iats.quantile(.5),
                'client_iat_25q': client_iats.quantile(.25),
                'client_iat_75q': client_iats.quantile(.75),

                'server_iat_mean': server_iats.mean(),
                'server_iat_median': server_iats.quantile(.5),
                'server_iat_25q': server_iats.quantile(.25),
                'server_iat_75q': server_iats.quantile(.75),
            })

        # return pd.Series(stats, index=stats.keys()).fillna(0)
        return stats


class FeatureTransformer:
    """
    fit_transform() processes raw targets and features pandas objects,
    learning new labels and scalers and one-hot encoding selected
    ones.

    IMPORTANT: USE load_transform() if you want to use
    TRAINED classifiers

    returns X_train, y_train, X_test, y_test
    """

    def __init__(self,
                 config,
                 categ_features=None):
        self._config = config
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.one_hot = OneHotEncoder()

        self._folder_pref = self._config['general']['classifiers_folder'] + os.sep
        self._one_hot_file = self._folder_pref + 'one_hot.dat'
        self._scaler_file = self._folder_pref + 'scaler.dat'
        self._le_file = self._folder_pref + 'le.dat'
        self.categ_features = categ_features or ['client_found_tcp_flags',
                                 'server_found_tcp_flags']
        self._split = float(self._config['offline']['splitRatio'])

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

    def _fit_transform_one_hot(self, features):
        selected = features[self.categ_features]
        one_hot = self.one_hot.fit_transform(selected).toarray()
        joblib.dump(self.one_hot, self._one_hot_file)
        return one_hot, features.drop(self.categ_features, axis=1)

    def _load_transform_one_hot(self, features):
        self.one_hot = joblib.load(self._one_hot_file)
        selected = features[self.categ_features]
        one_hot = self.one_hot.transform(selected).toarray()
        return one_hot, features.drop(self.categ_features, axis=1)

    def fit_transform(self, features, targets):
        one_hot_features, features = self._fit_transform_one_hot(features)

        F_tr, F_test, X_oh_tr, X_oh_test, t_tr, t_test = train_test_split(features,
                                                                          one_hot_features,
                                                                          targets,
                                                                          shuffle=True,
                                                                          test_size=self._split,
                                                                          stratify=targets)

        X_tr, y_tr = self._fit_transform_scale_and_labels(F_tr, t_tr)
        X_test, y_test = self._transform_scale_and_labels(F_test, t_test)

        return np.hstack([X_tr, X_oh_tr]), y_tr, np.hstack([X_test, X_oh_test]), y_test

    def load_transform(self, features, targets):
        one_hot_features, features = self._load_transform_one_hot(features)

        F_tr, F_test, X_oh_tr, X_oh_test, t_tr, t_test = train_test_split(features,
                                                                          one_hot_features,
                                                                          targets,
                                                                          shuffle=True,
                                                                          test_size=self._split,
                                                                          stratify=targets)

        X_tr, y_tr = self._load_transform_scale_and_labels(F_tr, t_tr)
        X_test, y_test = self._transform_scale_and_labels(F_test, t_test)

        return np.hstack([X_tr, X_oh_tr]), y_tr, np.hstack([X_test, X_oh_test]), y_test


def read_csv(config, csv_file=None):
    """
    process() removes rare protocols and flows, splits DataFrame
    into target vector and feature matrix 
    """

    if not csv_file:
        csv_file = os.path.join(config['offline']['csv_folder'],
                                config['parser']['csvFileTraining'])

    flow_features = pd.read_csv(csv_file,
                                sep='|',
                                index_col=0)


    # convert SSL_No_cert to SSL
    flow_features.replace('SSL_No_Cert', 'SSL', inplace=True)
    flow_features.replace('Unencrypted_Jabber', 'Jabber', inplace=True)
    
    flow_features.drop('subproto', axis=1, inplace=True)
    
    flow_features.fillna(0, inplace=True)
    
    # delete rarely occuring flows, identifying them first
    found_apps = flow_features.proto.value_counts()
    print(found_apps)
    apps_to_del = [app for app, value in
                   zip(found_apps.index, found_apps)
                   if value < int(config['parser']['minNumberOfFLowsPerApp'])]
    
    flow_features = flow_features[flow_features['proto'].map(lambda x: x not in apps_to_del)]
    
    result_features = flow_features.drop('proto', axis=1), flow_features['proto']


    return result_features