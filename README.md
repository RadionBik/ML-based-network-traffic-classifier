# Network traffic classifier based on statistical properties of application flows

UPDATE 18/03/2019: Refactored in OOP-style, more flexibility and features! 

UPDATE 23/05/2020: Replaced custom flow-parsing mechanism with NFStream

## Key features

* Configuration of ML algorithms and their parameter search space via the `classifiers_config.yaml` file.

* Automatic feature extraction and processing for ML algorithms.

* Support of various ML algorithms: Logistic regression, SVM, Decision Tree, Gradient Boosting, Random Forest, Multilayer Perceptron. 


## Project structure

* `flow_parser.py` converts a PCAP-file into a .csv file with features and labels (obtained from `NFStream`). 
The distinguishing feature is exporting of raw packet-features (e.g. 
packet/payload sizes, timestamps, various packet-fields) in a numpy array,
allowing such features as percentiles that are tricky to calculate in 
online mode. 

* `feature_processing.py` a wrapper of sklearn's transformers to 
ease selecting of used features.

* `datasets` contains utils to load/merge parsed `.csv` files, reassigns target classes (specific for my task).

* `report.py` contains a class for evaluation of the classifiers, plotting confusion matrices and scores. 

* `classifiers.py` contain the wrapper class ClassifierHolder that stores a model, its search space. 
Extending with other algorithms is possible by updating the REGISTRED_CLASSES variable.

* `evaluate_classifiers.py` contains an example pipeline when a dataset is ready.

* `pcap_files/` includes an example `.pcap` that is analyzed by the program modules by default. 
Also contains task-specific pre-processing script.

* `csv_files/` stores outputs of `flow_parser.py`, includes task-specific script.

## Module interfaces
### flow_parser.py

`flow_parser.py -p [--pcapfiles ...] -o [--output]`
* **-p --pcapfiles** - one or more PCAP-files to process
* **-o --output** - .csv file target to store the parsing results 

### evaluate_classifiers.py

`evaluate_classifiers.py -c [--config]`
* **-c --config** - the configuration `.yaml` to use, defaults to `classifiers_config.yaml`
* **--dataset** -- path to preprocessed `.csv` dataset (see the `dataset` package) 

## Usage example

1. A feature file has to be prepared before running model training, so make sure a `.csv` is already present 
by running `flow_parser.py`. 

2. OPTIONAL. Run `python dataset/formatter.py`. It creates a target column specified in `settings:TARGET_CLASS_COLUMN`,
 change the one if this step is omitted.

3. Make a copy of the `classifiers_config.yaml` to experiment with and
test classifiers via `evaluate_classifiers.py`

