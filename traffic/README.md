# What did I try?
I began with the requirements of the problem  that the `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)` and the output layer of the neural network should have NUM_CATEGORIES units. Initially using a single convolution layer and pooling layer inspired by lecture. From there, I began experimenting with all of the various configurations.

Notable Attempts:
- My initial attempt at mimicing handwriting.py from lecture resulted in around 5% accuracy.
- Adding additional convolution/pooling rounds brought this up to around 95%.
- Increasing the convolution kernel sizes drastically reduced accuracy and increased training time.
- Removal of flattening layers broke the network due to incompatible shapes.
- Trying to make the convolution kernals and pooling sizes proportionate to each other and to the input image width/height only slowed things down and reduced accuracy.
- Too many rounds on convolution and pooling slowed training times down
- I created a for loop to add many layers. I alternated between additional convolution, max-pooling, hidden layers, and dropouts. This drastically increased training time, but the accuracy improvements were neglible.

Also Tried:
- Different configurations of Python and TensorFlow. Installing/Re-Installing/Verifying PATH variables were correct. I ran into some Python/TensorFlow/Windows compatibility issues.
- Different activation functions
- Different loss types such as  during model compilation
- Different configurations of convolution layers and pooling
- Different times of flattening layers
- Different output unit sizes
- Different convulation kernel sizes
- Different pool sizes
- Different types of padding in layers

#  What worked well?
Ultimately, my best balance of accuracy percentages and training times came from using multiple rounds of convolution and pooling, flattening the dimensions, a hidden layer and dropout to prevent overfitting, and a "softmax" activation function in the output layer to build a probability distribution of which category each image is likely to be classified as. I created a for loop to add layers to the network which let me test many different network depths quickly.

- (3, 3) kernel size within the convolutional layers outperformed everything else that I tried.
- (2, 2) pool size outperformed everything else.
- "relu" activation functions outperformed "sigmoid" and others when I mixed them into various layers.

I couldn't break the 98% accuracy marker. Usually my best variations were between 97-98% on average.

#  What didn't work well?
Trying various filter kernal sizes within convlulational layers and different pooling sizes often didn't match up to correctly. It was also important to understand when to flatten else the output of a layer may not be the correct input shape for the next layer.

# What did I notice?
- Using larger kernel sizes in convolutation layers slowed down the training and reduced accuracy.
- Using too large of pooling sizes often didn't work at all due to shape and size constraints of outputs.
- Using too few units within layers resulted in low amounts of learning, though, the training times could be faster.
- It's sometimes hard to distinguish between minor changes since accuracy percentages will not always be the same.
- The data must contain a flattening layer else it cannot match the output format.
- Softmax activation function in the output layer performs better with respect to accuracy than sigmoid.
- Increasing the dropout rate to say, 0.95 just reduces its ability to learn and it seems to gain less insight between epochs resulting in lower accuracy.
- TensorFlow and overly complex networks crashed my computer frequently.
- Too large of unit sizes within layers leads to unnecessarily long training times, frequent crashes, and little additional accuracy.
- Too many interweaved layers of dropout caused significant reductions in accuracy.
- Interweaving hidden layers within the convolution and pooling layers reduced accuracy significantly, caused frequent crashes, and slowed training times.