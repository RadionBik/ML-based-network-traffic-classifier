
## Transformer-based network traffic generator and classifier 

Given currents trends in web-protocol development (e.g. eSNI, DNS-over-*),
 plain text information in traffic sessions is disappearing. In order
to classify the flows, one of few options is to use statistical discriminators
based on packet size (PS) and inter-packet time (IPT) features. 
Moreover, exactly the same features are usually produced by traffic 
flow generators. 
 
That gives an idea to develop a common neural network framework for 
creating statistical generators and classifiers. A reasonable choice
 can be Transformer architecture that showed SOTA on numerous NLP benchmarks.
Since we need a generative model, GPT-2 seems to be a good option to start 
with, luckily, `huggingface` did all the dirty stuff implementing it.

In order to use the models, the initial packet feature space (PS + IPT) 
has to be quantized into discrete sequences. I used K-Means for this 
purpose and given the expected dataset size (millions of flows),
the libKMCUDA's implementation was adopted to transform prior scaled 
 packet features into integer sequences of cluster numbers (see 
 `quantizer.py`).

Generative pretraining is a viable option to get a powerful classifier without
having much target data. We can pretrain the model in the following ways:
1. Using unlabeled data. Allows to further use the model as a feature
extractor for various classifiers (e.g linear, K-nn, uSVM) or to be completely
fine-tuned on a classification task.
2. Using labeled data. The model is trained with first sequence tokens 
denoting traffic class that afterwards allows to sample class-specific
packet clusters. Moreover, the same benefits as above are preserved.
 
Check the following link for pre-trained models and used datasets: 
```
http://51.77.194.175:9000/minio/traffic-classifier/
```

More details can be found in the following papers (please, cite the first one):
```
@article{Bikmukhamedov2021MultiClassNT,
  title={Multi-Class Network Traffic Generators and Classifiers Based on Neural Networks},
  author={R. Bikmukhamedov and A. Nadeev},
  journal={2021 Systems of Signals Generating and Processing in the Field of on Board Communications},
  year={2021},
  pages={1-7},
  url = {https://doi.org/10.1109/IEEECONF51389.2021.9416067}
}

@article{bikmukhamedov2020,
  author = {Bikmukhamedov, R. F. and Nadeev, A.F.},
  title = {Generative transformer framework for network traffic generation and classification},
  journal = {T-Comm},
  year = {2020},
  number = {11},
  vol = {14},
  pages = {64--71},
  url = {http://media-publisher.ru/wp-content/uploads/Nom-11-2020-s.pdf}
}
```
