import logging
import pathlib
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from pandarallel import pandarallel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from transformers import GPT2Model

from evaluation_utils.modeling import flows_to_packets
from flow_parsing.features import (
    FEATURE_PREFIX,
    FEATURE_FUNCTIONS,
    CONTINUOUS_NAMES,
    generate_raw_feature_names,
    calc_parameter_stats
)
from flow_parsing.utils import get_df_hash, save_dataset, read_dataset
from gpt_model.tokenizer import PacketTokenizer
from settings import TARGET_CLASS_COLUMN, DEFAULT_PACKET_LIMIT_PER_FLOW
from .utils import iterate_batch_indexes

logger = logging.getLogger(__name__)
pandarallel.initialize()


class BaseFeaturizer:
    def __init__(self, packet_num, consider_iat_features=True, target_column=TARGET_CLASS_COLUMN):
        self.target_encoder = LabelEncoder()
        self.target_column = target_column

        self.raw_features: List[str] = generate_raw_feature_names(
            packet_num,
            base_features=('packet', 'iat') if consider_iat_features else ('packet',)
        )

    def encode_targets(self, data: pd.DataFrame) -> np.ndarray:
        return self.target_encoder.transform(data[self.target_column])

    def fit_target_encoder(self, data: pd.DataFrame) -> np.ndarray:
        return self.target_encoder.fit_transform(data[self.target_column])

    def fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TransformerFeatureExtractor(BaseFeaturizer):
    def __init__(
            self,
            transformer_pretrained_path,
            packet_num,
            mask_first_token=False,
            reinitialize=False,
            device=None
    ):
        super().__init__(packet_num, consider_iat_features=True)
        assert packet_num > 0, 'raw packet sequence length must be > 0'
        self._pretrained_path = pathlib.Path(transformer_pretrained_path)
        self.tokenizer = PacketTokenizer.from_pretrained(transformer_pretrained_path,
                                                         flow_size=packet_num)
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        feature_extractor = GPT2Model.from_pretrained(transformer_pretrained_path).to(self.device)
        self.reinitialize = reinitialize
        if self.reinitialize:
            logger.info('resetting model weights')
            feature_extractor.init_weights()
        self.feature_extractor = feature_extractor.eval()
        self.mask_first_token = mask_first_token

    def _get_transformer_features(self, df, batch_size=1024):
        filename = (get_df_hash(df) +
                    self._pretrained_path.stem +
                    ('_mask_first' if self.mask_first_token else '') +
                    ('_reinitialize' if self.reinitialize else ''))
        tmp_path = pathlib.Path('/tmp') / filename
        if tmp_path.is_file():
            logger.info(f'found cached transformer features, loading {tmp_path}...')
            return read_dataset(tmp_path, True)

        logger.info(f'did not find cached transformer features at {tmp_path}, processing...')
        merged_tensor = np.empty((len(df), self.feature_extractor.config.hidden_size))
        for start_idx, end_idx in iterate_batch_indexes(df, batch_size):
            raw_subset = df[self.raw_features].iloc[start_idx:end_idx]
            encoded_flows = self.tokenizer.batch_encode_packets(raw_subset).to(self.device)
            if self.mask_first_token:
                encoded_flows['attention_mask'][:, 0] = 0
            with torch.no_grad():
                output = self.feature_extractor(**encoded_flows)
            output = output[0].to('cpu')  # last hidden state (batch_size, sequence_length, hidden_size)
            # average over temporal dimension
            output = output.mean(dim=1).numpy()
            merged_tensor[start_idx:end_idx, :] = output

        save_dataset(pd.DataFrame(merged_tensor), tmp_path)
        return merged_tensor

    def fit_transform_encode(self, data):
        X_feat = self._get_transformer_features(data)
        y = self.fit_target_encoder(data)
        return X_feat, y

    def transform_encode(self, data):
        X_feat = self._get_transformer_features(data)
        y = self.encode_targets(data)
        return X_feat, y


class Featurizer(BaseFeaturizer):
    """
    Featurizer processes features from a pandas object by merging results from scalers, one-hot encoders
    and encodes target labels
    """

    def __init__(self,
                 packet_num,
                 cont_features=None,
                 categorical_features=None,
                 consider_tcp_flags=True,
                 consider_j3a=True,
                 consider_raw_features=True,
                 consider_iat_features=False,
                 target_column=TARGET_CLASS_COLUMN):
        super().__init__(packet_num, consider_iat_features, target_column)

        self.column_converter = None
        if not consider_raw_features:
            self.raw_features = []
        self.consider_iat_features = consider_iat_features
        self.consider_tcp_flags = consider_tcp_flags
        self.consider_j3a = consider_j3a

        self.categorical_features = ['ip_proto'] if categorical_features is None else categorical_features
        self.cont_features = self._get_cont_features() if cont_features is None else cont_features
        self.try_extract_derivative_features = cont_features is None

        if self.consider_j3a:
            self.categorical_features.extend(['ndpi_j3ac', 'ndpi_j3as'])

        if self.consider_tcp_flags:
            self.categorical_features.extend([f'{FEATURE_PREFIX.client}found_tcp_flags',
                                              f'{FEATURE_PREFIX.server}found_tcp_flags'])

    def _get_cont_features(self):
        # here we expect features to be consistent with flow_parser's
        base_features = ['bulk', 'packet']
        if self.consider_iat_features:
            base_features.append('iat')

        cont_features = []
        for prefix in [FEATURE_PREFIX.client, FEATURE_PREFIX.server]:
            for derivative in list(FEATURE_FUNCTIONS.keys()):
                for base in base_features:
                    cont_features.append(prefix + base + derivative)
        return cont_features

    def _filter_non_existing_features(self, data: pd.DataFrame):
        data_features = set(data.columns)

        if set(self.raw_features) - data_features:
            found_features = list(set(self.raw_features) & data_features)
            logger.warning(f'skipping the following raw features: {set(self.raw_features) - data_features}')
            self.raw_features = found_features

        if set(self.cont_features) - data_features:
            found_features = list(set(self.cont_features) & data_features)
            logger.warning(f'skipping the following continuous features: {set(self.cont_features) - data_features}')
            self.cont_features = found_features

        if set(self.categorical_features) - data_features:
            found_features = list(set(self.categorical_features) & data_features)
            logger.warning(f'skipping the following categorical features: '
                           f'{set(self.categorical_features) - data_features}')
            self.categorical_features = found_features

    def _parse_derivatives_if_needed(self, data):
        if self.try_extract_derivative_features:
            data = self.calc_packets_stats_from_raw(data)
        return data

    def _fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ init transformers upon actual fitting to check for non-existing columns """
        data = self._parse_derivatives_if_needed(data)

        self._filter_non_existing_features(data)
        feature_set = []

        if self.cont_features:
            feature_set.append(("scaler", StandardScaler(),
                                self.cont_features))

        if self.categorical_features:
            feature_set.append(("one_hot", OneHotEncoder(handle_unknown='ignore', sparse=False),
                                self.categorical_features)),

        if self.raw_features:
            # TODO replace with PacketScaler
            feature_set.append(('raw_features', StandardScaler(), self.raw_features))

        self.column_converter = ColumnTransformer(feature_set)

        X_train = self.column_converter.fit_transform(data)
        y_train = self.fit_target_encoder(data)
        logger.info(f'{X_train.shape[0]} train samples with {self.n_classes} classes')
        return X_train, y_train

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self._fit_transform_encode(data)[0]

    def fit_transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self._fit_transform_encode(data)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        data = self._parse_derivatives_if_needed(data)
        X_test = self.column_converter.transform(data)
        return X_test

    def transform_encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        data = self._parse_derivatives_if_needed(data)
        X_test = self.column_converter.transform(data)
        y_test = self.encode_targets(data)
        return X_test, y_test

    @property
    def n_classes(self):
        return len(self.target_encoder.classes_)

    def calc_packets_stats_from_raw(self, data: pd.DataFrame):
        def calc_flow_packet_stats(flow: np.ndarray):
            subflow = flow[:2 * DEFAULT_PACKET_LIMIT_PER_FLOW]
            packets = flows_to_packets(subflow)
            from_idx = packets[:, 0] > 0
            to_idx = packets[:, 0] < 0

            stats = {}
            for direction, packet_idx in zip(
                    (FEATURE_PREFIX.server, FEATURE_PREFIX.client),
                    (from_idx, to_idx)
            ):
                try:
                    ps_derivatives = calc_parameter_stats(np.abs(packets[packet_idx, 0]), direction, 'packet')
                    stats.update(ps_derivatives)
                except ValueError:
                    continue

                if self.consider_iat_features:
                    try:
                        iat_derivatives = calc_parameter_stats(packets[packet_idx, 1], direction, 'iat')
                        stats.update(iat_derivatives)
                    except ValueError:
                        continue

            return stats

        if any(FEATURE_PREFIX.server + feature in data.columns for feature in CONTINUOUS_NAMES):
            logger.warning('packet stats has been found in dataframe, skipping calculation')
            return data

        tmp_path = pathlib.Path('/tmp') / (get_df_hash(data) + '_iat_' + str(self.consider_iat_features))
        if tmp_path.is_file():
            logger.info('found cached dataset version, loading...')
            return read_dataset(tmp_path, True)

        raw = data.filter(regex='raw_')
        packet_stats = raw.parallel_apply(calc_flow_packet_stats, axis=1, raw=True, result_type='expand').tolist()
        packet_stats = pd.DataFrame(packet_stats).fillna(0)
        logger.info('calculated the derivatives from raw features')
        data = data.join(packet_stats)
        save_dataset(data, save_to=tmp_path)
        return data
