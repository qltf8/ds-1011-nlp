# ds-1011-nlp
HW2

#1 RNN/CNN-based Natural Language Inference

Train RNN- and CNN-based models to tackle the Stanford Natural Language Inference (SNLI) task.

#2 Implementation

For the encoder, we want the following:
• For the CNN, a 2-layer 1-D convolutional network with ReLU activations will suffice. We can perform a max-pool at the end to compress the hidden representation into a single vector..<br />
• For the RNN, a single-layer, bi-directional GRU will suffice. We can take the last hidden state as the encoder output. (In the case of bi-directional, the last of each direction, although PyTorch takes care of this.).<br />


Perform tuning over at least two of the following hyperparameters:<br />
• Varying the size of the hidden dimension of the CNN and RNN<br />
• Varying the kernel size of the CNN<br />
• Experiment with different ways of interacting the two encoded sentences<br />
(concatenation, element-wise multiplication, etc)<br />
• Regularization (e.g. weight decay, dropout)<br />
