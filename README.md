# Speaker-Embeddings-Correlation-Pooling

This is the original implementation of the pooling method introduced in "Speaker embeddings by modeling channel-wise correlations" by T. Stafylakis, J. Rohdin, and L. Burget (Interspeech 2021), a result of the collaboration between Omilia - Conversational Intelligence and Brno University of Technology (BUT), which you may find [here](https://arxiv.org/pdf/2104.02571.pdf).

The code is in TensorFlow1 (TF1) but it should work with TF2 too. I only provide the code for creating the network and the required hyperparameters. The training
hyperparameters we used can be found in the paper.

The code is well-commented, at least the part and (hyper-)parameters required for the correlation pooling. 

Apart from the experiments provided in the paper, the code allows the user to:
(a) Combine standard statistics pooling with correlation pooling, by concatenating the two pooling layers into a single one, and
(b) Extract correlation pooling from outputs of all 4 internal ResNet blocks (aka stages) and concatenate them in the pooling layer.

The code can be more efficiently written using tensor-only operators. However, to fasilitate research we have implemented it using lists of tensors, e.g. after merging frequency bins to frequency ranges. Despite this inefficiency, we observe no differences between correlation pooling and standard stats pooling in training speed.

So, try it and let us know what you'll get!
Themos

