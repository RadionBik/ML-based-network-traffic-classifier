# Network traffic classifier based on statistical properties of application flows

UPDATE 18/03/2019: Refactored in OOP-style, more flexibility and features! 

UPDATE 23/05/2020: Replaced custom flow-parsing mechanism with NFStream

UPDATE 17/09/2020: Added pytorch classifiers, including transformer-based one

## Key features

* Configurable feature extraction from network flows via `NFStream`.

* Possibility to test arbitrary sklearn algorithms (e.g. SVM, Random Forest, 
etc.) and configure their parameter search space via `.yaml` configs.

* Basic examples of pytorch classifiers and new generative transformer
framework that can be used for building traffic generators and 
classifiers.

* Option for experiment tracking with Neptune.

## Project structure

* `flow_parsing` contains scripts for parsing flow features and labels
from `.pcap` into `.csv` via `NFStream`. It can be
 used for exporting raw per-flow packet-features (e.g. packet/payload 
 sizes, timestamps, various packet-fields) in a numpy array, as well as
 derivative statistics, such as feature percentiles, etc.

* `evaluation_utils` contains utilities for evaluation of traffic 
classifiers and generators.

* `sklearn_classifiers` contains wrapper for sklearn-like classifiers 
and example pipeline script. Used models and their parameters are specified
via the `.yaml` configuration file. Check and modify `utils.py:REGISTERED_CLASSES` 
to support the needed models.

* `nn_classifers` includes base class for pytorch-lightning classifier and
some basic derivatives.

* `gpt_model` has all the code required for building your own 
transformer-based traffic generator and classifier, along with a link to 
model checkpoints. See the package for more info.

## Usage example for sklearn-based classifiers

1. A feature file has to be prepared before running model training, so 
make sure to create a `.csv` dataset by running, for example:
 
    ```PYTHONPATH=. python flow_parsing/pcap_parser.py -p flow_parsing/static/example.pcap --online_mode``` 

2. OPTIONAL. Postprocess parsed `.csv` as needed, e.g. split into train-test,
reassign target columns.

3. Create own version of `config.yaml` to experiment with and
test classifiers:

    ```
   PYTHONPATH=. python sklearn_classifiers/run_training.py 
        --train_dataset csv_files/example_20packets.csv 
        --target_column ndpi_category 
        --continuous 
   ```

