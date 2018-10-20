# ds-1011-nlp
HW1

#1 Bag of Word Model

The bag of word model sums all of the words’ embedding in an example and then gets the average which is the vector’s representation of the example. Finally, the vector is converted to a scalar and scale it between 0 and 1 by sigmoid function representing the possibility that the example is positive.

#2 Implementation

Since the task was to do hyper-parameter search, I built a class which integrated the construction of vocabulary, model training, model selection, and model evaluation. Basically, the class did the following 6 tasks.<br />
• Use torchtext.Field to process text data, and build vocabulary and word look-up table.<br />
• Use torchtext.data.TabularDataset to load data from csv file and split the data into train, validation and test set. The number of training examples is equal to 20000, validation examples is 5000, and the test examples is 25000. Since the split function use stratified sampling, each example set contains approximately the same percentage of samples of each target class as the complete set. In this task, the proportion of positive examples and negative examples is equal in train, validation, and test examples.<br />
• Use torchtext.data.BucketIterator to define an iterator that loads batches of data from a dataset. The benefit of BucketIterator is that this method can batch examples of similar lengths together instead of setting a fix length for all of the examples.<br />
• Train the model on training dataset. For each epoch, the model will be evaluated on validation examples in term of accuracy rate and AUC. We will keep track of the model having the best accuracy rate on validation examples. And if the best accuracy rate doesn’t improve in the last several epochs, the training process will stop. And draw the training curve showing how the accuracy rate is changed after each epoch for training examples and validation examples respectively.<br />
• Apply the best model in terms of accuracy rate on validation examples to test examples, and calculate the accuracy rate and draw the confusion matrix.<br />
• Show 3 correct and 3 incorrect predictions of the model on the validation set. There are total 9 hyperparameters for the class:<br />
• n gram: an N-token sequence of words<br />
• min freq: the minimum frequency needed to include a token in the vocabulary. This parameter is used<br />
to control vocabulary size<br />
• embedding dim: the vector size for each word<br />
• optimizer: optimization algorithms. Adam or Adagrad or SGD<br />
• learning rate: step size<br />
• weight decay: weight decay (L2 penalty)<br />
• decay rate: the rate of the learning rate reducing (via torch.optim.lr scheduler.LambdaLR)<br />
• batch size: batch size<br />
• early stopping rounds: will stop training if one metric of one validation data doesn’t improve in last early stopping round rounds<br />
       
