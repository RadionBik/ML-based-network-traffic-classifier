## FS-NET

Reimplementation of FS-NET model without reconstruction loss, 
which harmed the performance according to the reported results in the original 
paper:

```
@inproceedings{LiuHXCL19,
  author    = {Chang Liu and
               Longtao He and
               Gang Xiong and
               Zigang Cao and
               Zhen Li},
  title     = {FS-Net: {A} Flow Sequence Network For Encrypted Traffic Classification},
  booktitle = {{IEEE} Conference on Computer Communications (INFOCOM), 2019},
  pages     = {1171--1179},
  year      = {2019}
}
```
From my point of view, there is some inconsistency between the paper's 
description and implementation found in https://github.com/WSPTTH/FS-Net, 
particularly regarding the presence of Selu activation for the final 
output in eq. 17.

As a bonus, the training script has 2 options for the model's input: 
 (i) either packet size sequences (as in the paper), or (ii) K-Means centroids 
 for (PS, IPT) features (similarly to the transformer model in this repo). 