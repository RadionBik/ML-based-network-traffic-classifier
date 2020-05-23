# Network traffic classifier based on statistical properties of application flows

UPDATE 18/03/2019: Refactored in OOP-style, more flexibility and features! 

UPDATE 23/05/2020: Replaced custom flow-parsing mechanism with NFStream

## Key features

* Configuration of testing scenarios via the `config.ini` file.

* Automatic feature extraction and processing for ML algorithms.

* Support of various ML algorithms: Logistic regression, SVM, Decision Tree, Gradient Boosting, Random Forest, Multilayer Perceptron. 

* Automatic optimization of the algorithms' parameters (can be activated in the `config.ini` file).

* Setting ML algorithm's settings via `classifiers.yaml`

## Project structure

* `flow_parser.py` converts collected PCAP-files into a .csv file with features and labels (obtained from `NFStream`).

* `feature_processing.py` contains loaders for parsed .csv files, with automatic cleaning up of `.csv` files, e.g. remove seldom flows, etc.

* `report.py` contains a class for evaluation of the classifiers, plotting confusion matrices and scores. WARNING: may be broken, subject for fixing in future releases.

* `classifiers.py` contain the wrapper class ClassifierEnsemble that allows for parameter optimization, training, restoring and testing of the ML algorithms with sklearn-like interface. The parameter search space is specified during initialization. *Extending with other algorithms is encouraged.*

* `traffic_classifier.py` runs the ML pipeline when a .csv is present.

* `pcap_files/` includes an example .pcap that is analyzed by the program modules by default.

* `trained_classifiers/` folder is used for storage of trained classifier that can be used later for validations on different traffic or in the live mode.

* `csv_files/` stores outputs of `flow_parser.py` .

* `figures/` this is where the output from the ClassifierEvaluator class is produced.  

## Module interfaces
### flow_parser.py

`flow_parser.py -p [--pcapfiles ...] -o [--output]`
* **-p --pcapfiles** - one or more PCAP-files to process
* **-o --output** - .csv file target to store the parsing results 

### traffic_classifier.py

`traffic_classifier.py -c [--config]`
* **-c --config** - the configuration file to use
* **--load-processors** -- overrides the settings in the configuration file, loads custom feature preprocessors 
* **--fit-processors** -- overrides the settings in the configuration file, fits new feature preprocessors 
* **--load-classifiers** -- overrides the settings in the configuration file, loads custom classifier models
* **--fit-classifiers** -- overrides the settings in the configuration file, fits new classifier models 

*At this stage, you are encouraged to use Jupyter Notebook to experiment with this repo's features, especially with the `classifiers.py` module. Many things are hardcoded in order to suite tasks I solved at the moment of writing this software.*

## First run

Make a copy of the `config.example.ini` to experiment with. If the config file during a module's start is not provided, `config.ini` is looked over by default.

A feature file has to be prepared before running model training, so make sure a .csv is already present by running `flow_parser.py`.   
