# Report for Graded Assignment 1
> CS-461 Foundation Models and Generative AI

## Methods
For this assignment, the SimCLR self-supervised algorithm was implemented for the classification of 64x64 images coming from a downsampled subset of [ImageNet-1k dataset](https://www.image-net.org/). SimCLR was chosen as it is the best performing algorithm of those covered in Notebook 2 (Week 3), it is very simple, and also does not suffer from collapsing issues, given that it uses negative samples in the training loss as contrastive examples.

The baseline is the same as the one given in Notebook 2, adapted to get features of 1000 dimensions. It uses a slightly modified ResNet-18 architecture (first layer has a convolution with kernel size 3 and no max pooling) to output the features, which are then sent through a 2-layer MLP with dimensions 1000 (input), 2048 (hidden), 128 (output). The batch size was set to 256, the highest that fits in memory, given that contrastive methods and in particular SimCLR benefit from large batch sizes if the number of epochs is small (Chen et al., 2020). The rest of the setup was kept identical to Notebook 2 (50 epochs, cosine scheduler with warmup, with a learning rate of 0.6)

A first experiment was done with a larger model (ResNet-34 instead of ResNet-18, so roughly 2x more parameters), keeping the rest of the setup the same. Performance was very similar to the baseline, which led to the insight that model size was not the bottleneck, but rather, given that accuracies plateau after 50 epochs (see plots in the `CS461_Assignment1.ipynb` notebook), it means that the model has essentially saturated what it can learn from the available data. However, we are limited to using only the training subset of 100k images (200 classes). One way to circumvent this data issue, is to simply do more data augmentation. The hypothesis is that using only 2 augmented views per data point (as done in the baseline SimCLR) does not fully exploit the learning signal that could be extracted from the data. In other words, with such a limited dataset size, increasing the diversity or number of augmented views per image could help the model see more varied examples of the same underlying content, effectively acting as a form of data multiplication. This would allow the model to continue improving its representations even when the amount of raw data is fixed.

Thus, to improve performance, we propose the following changes over the baseline:

1. Change the architecture from ResNet-18 to ResNet-34 to improve the expressive power of the encoder network. On one hand, this increases the number of parameters from ~14M to ~24M (thus roughly doubling model size), but it also changes the first convolution's kernel from 3x3 to 7x7, which increases the receptive field and may further enhance the model's capacity to capture spatial patterns and richer representations.
2. Multiply by 4 the amount of training data, by generating 8 augmented views per data point instead of just 2. Note that we still need to have views always paired two by two in a given batch, and never have more than 2 views from the same data point per batch, so that we can still use the SimCLR loss without modifications.

## Results

| Metric        | Baseline | Improved (2x parameters, 4x data) |
|----------------|-----------|-----------|
| k-NN accuracy  | 0.18      | 0.30      |
| Linear accuracy | 0.20      | 0.32      |
| OOD k-NN accuracy | 0.12   | 0.19      |
| OOD Linear accuracy | 0.19 | 0.24      |

A few comments can be made. First, we see that for this SimCLR task, in this model size / dataset size regime, some kind of scaling law seems to apply. This is interesting and would require further investigation to better understand the details of this law. Second, we see that OOD performance do improve, although not as well as in distribution performance. Even if the performance jump is smaller, the fact that performance increases is good news, since it means that our model could still learn at least some useful representation of the data, that generalizes to unseen classes. Finally, we can conclude that our initial hypothesis that 2 augmented views per data point would not fully exploit the learning signal available in the data seems to hold, as we could improve k-NN performance by a factor of 1.5 simply by increasing the number of views per data point.
