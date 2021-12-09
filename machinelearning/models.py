import nn
import numpy as np
import matplotlib.pyplot as plt

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        result = self.run(x)
        scalar_result = nn.as_scalar(result)
        if scalar_result >=0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        #iterating like shown in the spec
        batch_size = 1

        #assuming that the data will always have something that is missclassified
        # converged = True

        while True:
            converged =True

            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    converged = False
                    #https://inst.eecs.berkeley.edu/~cs188/fa21/assets/slides/lec21.pdf#page=16
                    #following the instructions on this slide to properly use the update function
                    # if nn.as_scalar(self.w) * self.get_prediction(x) >= 0:
                    #     y_star = 1
                    # else: 
                    #     y_star = -1

                    # direction = self.get_prediction(x)

                    # self.w.update(x, nn.as_scalar(y))
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))

            if converged: 
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        
        "*** YOUR CODE HERE ***"
        # RECOMMENDED VALS FOR HYPERPARAMETERS
        # Hidden layer sizes: between 10 and 400.
        # Batch size: between 1 and the size of the dataset. 
                # For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
        # Learning rate: between 0.001 and 1.0.
        # Number of hidden layers: between 1 and 3.
        self.hidden_layer_size = 60
        self.batch_size = 100
        self.learning_rate = 0.01
        
        # FROM PIAZZA: Since the size of the input is (batch_size x 1), the W_1 dimension should be (1 x hidden_layer_size)

        # parameter matrices W_1 and W_2
        self.W_1 = nn.Parameter(1, self.hidden_layer_size)
        self.W_2 = nn.Parameter(self.hidden_layer_size, 1)

        # parameter vectors b_1 and b_2
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, 1)
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x dot W_1 + b_1) dot W_2 + b_2
        
        myW_1 = nn.Linear(x, self.W_1)
        relu = nn.ReLU(nn.AddBias(myW_1, self.b_1))
        myW_2 = nn.Linear(relu, self.W_2)
        y_pred = nn.AddBias(myW_2, self.b_2)
        return y_pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return nn.SquareLoss(y, y_pred)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            # FROM PIAZZA: update() is tuned to do gradient ASCENT
            # so put in negative multiplier
            loss = self.get_loss(x, y)
            grad_W_1, grad_b_1, grad_W_2, grad_b_2 = nn.gradients(loss, [self.W_1, self.b_1, self.W_2, self.b_2])

            self.W_1.update(grad_W_1, -self.learning_rate)
            self.b_1.update(grad_b_1, -self.learning_rate)
            self.W_2.update(grad_W_2, -self.learning_rate)
            self.b_2.update(grad_b_2, -self.learning_rate)

            # print(nn.as_scalar(loss))
            if nn.as_scalar(loss) <= 0.015:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # RECOMMENDED VALS FOR HYPERPARAMETERS
        # Hidden layer sizes: between 10 and 400.
        # Batch size: between 1 and the size of the dataset. 
                # For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
        # Learning rate: between 0.001 and 1.0.
        # Number of hidden layers: between 1 and 3.
        self.hidden_layer_size = 300
        self.batch_size = 100
        self.learning_rate = 0.01

        # Since the size of the input is (batch_size x 784), the W_1 dimension should be (784 x hidden_layer_size)
        # W_2 dimension should be (hidden_layer_size x 10)
        #need to multiply input by (__ x 10) matrix to get output correct dimension
        self.W_1 = nn.Parameter(784, self.hidden_layer_size)
        self.W_2 = nn.Parameter(self.hidden_layer_size, 10)

        self.b_1 = nn.Parameter(1, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        myW_1 = nn.Linear(x, self.W_1)
        relu = nn.ReLU(nn.AddBias(myW_1, self.b_1))
        myW_2 = nn.Linear(relu, self.W_2)
        y_pred = nn.AddBias(myW_2, self.b_2)
        return y_pred
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return nn.SoftmaxLoss(y, y_pred)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)
            grad_W_1, grad_b_1, grad_W_2, grad_b_2 = nn.gradients(loss, [self.W_1, self.b_1, self.W_2, self.b_2])

            self.W_1.update(grad_W_1, -self.learning_rate)
            self.b_1.update(grad_b_1, -self.learning_rate)
            self.W_2.update(grad_W_2, -self.learning_rate)
            self.b_2.update(grad_b_2, -self.learning_rate)

            if dataset.get_validation_accuracy() >= 0.98:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 60
        self.batch_size = 100
        self.learning_rate = 0.01

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
