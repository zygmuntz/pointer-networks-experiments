# Pointer networks experiments

Code for genereting data, training and testing. There are two tasks: ordering single numbers, or sums or numbers (the `_sums` suffix).

The training scripts save model weights each epoch. When started, they attempt to load appropriate weights - corresponding to the hidden size and the chosen number of steps in a sequence.

This software builts upon [https://github.com/keon/pointer-networks](https://github.com/keon/pointer-networks).

Using Keras 2? [Try the keras-2.0 branch](https://github.com/zygmuntz/pointer-networks-experiments/tree/keras-2.0).
