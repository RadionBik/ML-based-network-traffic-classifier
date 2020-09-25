
## Transformer-based network traffic generator and classifier 

Given currents trends in web-protocol development (e.g. eSNI, DNS-over-*),
 plain text information in traffic sessions is disappearing. In order
to classify the flows, one of few options is to use statistical discriminators
based on packet size (PS) and inter-packet time (IPT) features. 
Moreover, exactly the same features are usually produced by traffic 
flow generators. 
 
That gives an idea to develop a common neural network framework for 
creating statistical generators and classifiers. A reasonable choice
 can be Transformer architecture that has shown SOTA on numerous NLP benchmarks.
Since we need a generative model, GPT-2 seems to be a good option to start 
with, luckily, `huggingface` did all the dirty stuff implementing it.

In order to use the models, the initial flow feature space (PS + IPT) 
has to be quantized into discrete sequences. K-Means is often used
for this purpose and given the expected dataset size (millions of flows),
 I used libKMCUDA's implementation to transform prior scaled 
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

More info is coming soon!