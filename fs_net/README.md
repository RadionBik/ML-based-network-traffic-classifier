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
description and implementation found in:

https://github.com/WSPTTH/FS-Net

particularly regarding the presence of Selu activation for the final 
output in eq. 17.

Moreover, current training script doesn't use only packet features as described
in the paper, but rather uses K-Means centroids for (PS, IPT) features, which
can is usually beneficial to the final performance due to presence of the
 timing information. 