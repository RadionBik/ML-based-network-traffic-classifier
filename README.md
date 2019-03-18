# Network traffic classifier based on statistical properties of application flows

UPDATE 18/03/2019: Refactored in OOP-style, more flexibility and features! 

## Key features

* Configuration of testing scenarios via the `config.ini` file.

* Automatic feature extraction and processing for ML algorithms.

* Support of various ML algorithms: Logistic regression, SVM, Decision Tree, Gradient Boosting, Random Forest, Multilayer Perceptron. 

* Automatic optimization of the algorithms' parameters (can be activated in the `config.ini` file).

Depends on dpkt, numpy, sklearn, pandas libraries for Python 3.x 

## Project structure

* `pcap_parser.py` contains the class that converts collected PCAP-files into a .csv file with features and labels (obtained from `nDPI`). WARNING: the processing is slow and takes x2 RAM of a `.pcap` size.

* `feature_processing.py` contains classes for feature crafting and cleaning of `.csv` files, e.g. remove seldom flows, etc.

* `config_loader.py` has a simple shared class for `config.ini` loading.  

* `report.py` contains a class for evaluation of the classifiers, plotting confusion matrices and scores.

* `classifiers.py` contain a wrapper class Traffic_Classifiers that allows for easy parameter optimization, training, restoring and testing of the ML algorithms with sklearn-like interface. The parameter search space is specified during initialization. *Extending with other algorithms is encouraged.*

* `ndpiReader_xxx` are binaries of the `nDPI v.2.9(?)` library compiled for Ubuntu 16.04 and recent Manjaro (as of start of 2019).

* `trained_classifiers/` folder is used for storage of trained classifier that can be used later for validations on different traffic or in the live mode.

* `csv_files/` stores outputs of `pcap_parser.py` .

* `figures/` this is where the output from the Classifier_Evaluator class is produced.  

## Module interfaces
### pcap_parser.py

`pcap_parser.py -p [--pcapfiles ...] -c [--config]`
* **-p --pcapfiles** - one or more PCAP-files to process
* **-c --config** - the configuration file to use

### classifiers.py

`classifiers.py -c [--config]`
* **-c --config** - the configuration file to use

*At this stage, you are encouraged to use Jupyter Notebook to experiment with this repo's features, especially with the `classifiers.py` module. Many things are hardcoded in order to suite tasks I solved at the moment of writing this software.*
