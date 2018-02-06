# Network traffic classifier based on statistical properties of application flows

Inspired by vnetserg/traffic-v2, although significantly rewritten and complemented.

##Key differences:

* Added configuration of testing scenarios via the 'config.ini' file

* Added support of unidirectional flows(on the cost of some stat. features)

* Added various ML algorithms: Logistic regression, SVM, Decision Tree, Gradient Boosting, Random Forest, Multilayer Perceptron.

* Added automatic optimization of the algorithms' parameters (can be enabled in the *config.ini* file)

* Update of the ndpiReader binary to 2.1.0, compiled for kernel 4.4.x

* Ported to Python3.6

To run the code, the dpkt, numpy, sklearn and pandas libraries for Python 3.x need to be installed. 

The *py/* folder has the following structure:
* *pcaptocsv.py* converts collected PCAP-files into a .csv file with features and labels
* *post_process_csv.py* performs further processing of a .csv file, e.g. cleaning rare occurring flows, etc.
* *classifier.py* is used for training, optimizing, testing the ML algorithms and printing reports.
* *ndpiReader* is a binary of the nDPI traffic analyzer

The *trained_classifiers* folder is used for storage of trained classifier that can be used later via adjusting the configuration file.

The *csv_files* folder stores the output of *pcaptocsv.py*

The *figures* folder contains the .pdf files with confusion matrices and the accuracy scores of classifiers.  

## The module interfaces
### pcaptocsv.py

`pcaptocsv.py [-o OUTPUT] [-s N] file [file ...]`
* **file [file ...]** - one or more PCAP-files
* **-o OUTPUT** - a name  for the output CSV-file ("flows.csv" by default)
* **-s N** - build the feature table only with first N segments of the transport layer (doesn't account for TCP handshake)

### classifier.py

It can be widely configured within the *config.ini* file.

The program prints to stdout fitting time, Accuracy, F-measure and Confusion Matrix (CM) of each tested algorithm.

The plots for CM and the accuracy score can be optionally saved to *.pdf* files (see *config.ini*)
