# MNIST digit recognizer implemented in numpy from scratch
This shows the basic workflow of a machine learning algorithm using a simple feedforward neural net. The derivative at the backpropagation stage is computed explicitly through the chain rule.

* The model is a 3-layer feedforward neural network (FNN), in which the input layer has 784 units, and the 256-unit hidden layer is activated by ReLU, while the output layer is activated by softmax function to produce a discrete probability distribution for each input.
* The training is through a standard gradient descent with step size adaptively changing by Root Mean Square prop (RMSprop).

## Results
```
loss after 1 iterations is 7.68388
training accuracy after 1 iterations is 14.1238%
loss after 501 iterations is 0.19777055
training accuracy after 501 iterations is 94.0643%
loss after 1001 iterations is 0.072008964
training accuracy after 1001 iterations is 97.9452%
final loss is 0.071490053
final training accuracy is 97.9548%
```
