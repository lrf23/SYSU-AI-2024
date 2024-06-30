import nn

### Question 1
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
        return nn.DotProduct(self.w,x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        res=nn.as_scalar(nn.DotProduct(self.w,x))
        if (res>=0) :
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Useful function:
        update(self, direction, multiplier) in class Parameter in nn.py
        retrieve seft.data in nn.Constant, e.g., if x is an object of nn.Constant, get its data via x.data
        """
        "*** YOUR CODE HERE ***"
        flag=False
        while ~flag:
            flag=True
            for x,y in dataset.iterate_once(1):
                y_p=self.get_prediction(x)
                if y_p!=nn.as_scalar(y):
                    self.w.update(x,nn.as_scalar(y))
                    flag=False
            if (flag):
                break
        

### Question 2
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer_size=20
        self.layers=3
        self.W=[]
        self.b=[]
        for i in range(self.layers):
            if i==0:
                self.W.append(nn.Parameter(1,self.layer_size))
                self.b.append(nn.Parameter(1,self.layer_size))
            elif i==self.layers-1:
                self.W.append(nn.Parameter(self.layer_size,1))
                self.b.append(nn.Parameter(1,1))
            else:
                self.W.append(nn.Parameter(self.layer_size,self.layer_size))
                self.b.append(nn.Parameter(1,self.layer_size))
            #self.batch_size=10

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        input_data=x
        for i in range(self.layers):
            if i==self.layers-1:
                return nn.AddBias(nn.Linear(input_data,self.W[i]),self.b[i])
            input_data=nn.ReLU(nn.AddBias(nn.Linear(input_data,self.W[i]),self.b[i]))
        

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
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size=100
        learning_rate=0.05

        
        while 1:
            flag=False
            for x,y in dataset.iterate_once(batch_size):
                loss=self.get_loss(x,y)
                g_wb=nn.gradients(loss,self.W+self.b)
                for i in range(self.layers):
                    self.W[i].update(g_wb[i],-learning_rate)
                    self.b[i].update(g_wb[self.layers+i],-learning_rate)
                if (nn.as_scalar(loss)<0.01) :
                    flag=True
            if (flag) :
                break

### Question 3
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
        self.layer_size=[300,100,10]
        self.layers=3
        self.W=[]
        self.b=[]
        for i in range(self.layers):
            if i==0:
                self.W.append(nn.Parameter(784,self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))
            else:
                self.W.append(nn.Parameter(self.layer_size[i-1],self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))
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
        input_data=x
        for i in range(self.layers):
            if i==self.layers-1:
                return nn.AddBias(nn.Linear(input_data,self.W[i]),self.b[i])
            input_data=nn.ReLU(nn.AddBias(nn.Linear(input_data,self.W[i]),self.b[i]))

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
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size=500
        learning_rate=0.2
        while 1:
            flag=False
            for x,y in dataset.iterate_once(batch_size):
                loss=self.get_loss(x,y)
                g_wb=nn.gradients(loss,self.W+self.b)
                for i in range(self.layers):
                    self.W[i].update(g_wb[i],-learning_rate)
                    self.b[i].update(g_wb[self.layers+i],-learning_rate)
                v_acc=dataset.get_validation_accuracy()
                print(v_acc)
                if (v_acc>=0.97) :
                    flag=True
            if (flag) :
                break




### Question 4
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
        self.layer_size=[800,800,5]
        self.layers=3
        self.W=[]
        self.b=[]
        for i in range(self.layers):
            if i==0:
                self.W.append(nn.Parameter(self.num_chars,self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))
            else:
                self.W.append(nn.Parameter(self.layer_size[i-1],self.layer_size[i]))
                self.b.append(nn.Parameter(1,self.layer_size[i]))

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
        input_data=xs
        hid_state=nn.Linear(xs[0],self.W[0])
        for x in xs[1:]:
            p1=nn.AddBias(nn.Linear(x,self.W[0]),self.b[0])
            p2=nn.AddBias(nn.Linear(hid_state,self.W[1]),self.b[1])
            hid_state=nn.ReLU(nn.Add(p1,p2))
        y_p=nn.AddBias(nn.Linear(hid_state,self.W[2]),self.b[2])
        return y_p
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
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size=100
        learning_rate=0.2
        v_acc=0
        while 1:
            flag=False
            for xs,y in dataset.iterate_once(batch_size):
                loss=self.get_loss(xs,y)
                g_wb=nn.gradients(loss,self.W+self.b)
                for i in range(self.layers):
                    self.W[i].update(g_wb[i],-learning_rate)
                    self.b[i].update(g_wb[self.layers+i],-learning_rate)
                v_acc=dataset.get_validation_accuracy()
                if (v_acc>=0.84) :
                    flag=True
            print(v_acc)
            if (flag) :
                break

