# Convolutional Neural Network
## Table of Content
### `composite` & `layer`
The Python module `composite` and `layer` implement various convolutional layer using pure numpy. However, the efficiency
of performing convolution through iteration in Python is extremely poor. They are probably more useful if I had
implemented them in C or using gonum in Golang. Anyways, they are good reference for back-propagation algorithms in
various layers.

### Datasets
Dataset folder has been git ignored. However there is a bash script for obtaining the CIFAR 10 data. There are two
datasets I am using in this project; CIFAR 10 for general practicing and Kaggle's dog-vs-cat for prototyping my pet
recognition project.

### `.ipynb`
Jupyter notebooks are primarily for notes and mathematic derivations.

### Tensorflow - `tf_xxx_model`
These are the actual models implemented using tensorflow.


## Quick Notes
### Padding `VALID` vs `SAME`
`VALID` means no padding, only drops the right most column or bottom most rows.
```
inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
               |________________|                dropped
                              |_________________|
```

`SAME` tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right.
```
            pad|                                      |pad
inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
            |________________|
                           |_________________|
                                          |________________|
```

## Architecture
AlexNet was implemented using

1. Input: `(N, 227, 227, 3)`
2. Convolution: `(N, 55, 55, 96)` using 96 11x11 filters with`stride=4` and `padding=0`
3. Max Pooling: `(N, 27, 27, 96)` using 3x3 filter with `stride=2`
4. Normalization: `(N, 27, 27, 96)` using batch normalization, normalize across channels
5. Convolution: `(N, 27, 27, 256)` using 256 5x5 filters with `stride=1` and `padding=2`
6. Max Pooling: `(N, 13, 13, 256)` using 3x3 filter with `stride=2`
7. Normalization: `(N, 13, 13, 256)` using batch normalization, normalize across channels
8. Convolution: `(N, 13, 13, 384)` using 384 3x3 filters with `stride=1` and `padding=1`
9. Convolution: `(N, 13, 13, 384)` using 384 3x3 filters with `stride=1` and `padding=1`
10. Convolution: `(N, 13, 13, 256)` using 256 3x3 filters with `stride=1` and `padding=1`
11. Max Pooling: `(N, 6, 6, 256)` using 3x3 filter with `stride=2`
12. Fully Connected: `(N, 4096)`
13. Fully Connected: `(N, 4096)`
14. Output: `(N, 1000)` which is the class score

It used heavy data augmentation and dropout rate of 0.5, which is not included above. The batch size was 128 and used
stochastic gradient descent with momentum 0.9. Learning rate was 1e-2 and reduced by factor of 10 manually when validation
accuracy plateaus. L2 weight decay was 5e-4.
