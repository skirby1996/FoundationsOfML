import numpy as np
import math

def hinge_loss(train_y, pred_y):
    loss = 0
    for i in range(train_y.size):
        loss += max(0, 1 - train_y[i] * pred_y[i])
    return loss
	
def squared_loss(train_y, pred_y):    
    loss = 0
    for i in range(train_y.size):
        loss += (train_y[i] - pred_y[i])**2
    return loss
	
def exponential_loss(train_y, pred_y):
    loss = 0
    for i in range(train_y.size):
        loss += math.exp(-1 * train_y[i] * pred_y[i])
    return loss
	
def l1_reg(w):
    return np.sum(np.absolute(w))
	
def l2_reg(w):
    return np.sum(np.square(w))

def shuffle_arrays(a, b):
    if a.shape[0] != b.shape[0]:
        print("Error: Arrays must have same size on axis 0")
        return None, None
    
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

def train_classifier(train_x, train_y, learn_rate, loss, lambda_val, regularizer, verbose=False):   
    # Initialize weights
    weights = np.random.random(train_x.shape[1])
    weights = weights * 2 - 1

    # Number of times to iterate through the whole dataset
    num_epochs = 1000

    # Number of items in training batch
    batch_size = 64

    if verbose:
        print("Training for %d epochs with batch size %d" % (num_epochs, batch_size))

    for epoch in range(num_epochs):
        # Shuffle arrays each epoch
        train_x, train_y = shuffle_arrays(train_x, train_y)

        for batch in range(train_y.size // batch_size):
            # Create batch
            start_ix = batch * batch_size
            stop_ix = (batch + 1) * batch_size
            batch_x = train_x[start_ix:stop_ix]
            batch_y = train_y[start_ix:stop_ix]

            # Compute average gradients for batch 
            orig_loss = loss(batch_y, test_classifier(weights, batch_x))
            if regularizer is not None:
                orig_loss += lambda_val * regularizer(weights)

            h = 0.0001
            dw = np.zeros(weights.size)
            for w in range(weights.size):
                weights[w] += h
                pred_y = test_classifier(weights, batch_x)
                new_loss = loss(batch_y, pred_y)
                if regularizer is not None: 
                    new_loss += lambda_val * regularizer(weights)
                dw[w] = (new_loss - orig_loss) / h
                weights[w] -= h
         
            dw /= batch_size
            weights -= learn_rate * dw

        if verbose:
            pred_y = test_classifier(weights, train_x)
            end_loss = loss(train_y, pred_y)
            if regularizer is not None:
                end_loss += lambda_val * regularizer(weights)
            end_loss /= train_y.size

            prog = int(20 * ((epoch + 1) / num_epochs))
            prog_bar = "[%s%s%s]" % ('=' * prog, ('=' if prog == 20 else '>'),  ' ' * (20 - prog))
            print("%06d/%06d - %s\tloss: %f" % (epoch + 1, num_epochs, prog_bar, end_loss), end="\r", flush=True)
    if verbose:
        print("\n")

    return weights

def test_classifier(w, test_x):
    pred_y = np.zeros(test_x.shape[0])
    for i in range(pred_y.size):  
        pred_y[i] = np.matmul(w, test_x[i]) 
     
    return pred_y

def compute_accuracy(test_y, pred_y):
    correct = 0
    for i in range(test_y.size):
        if test_y[i] * pred_y[i] > 0:
            correct += 1

    return correct / test_y.size

def get_id():
    return "tug31177"

def main():
    # Load dataset
    data = np.genfromtxt('winequality.csv', delimiter=',', skip_header=1) 

    # Parse dataset
    x = data[:, :-1]
    y = data[:, -1] 

    x_max = np.full(x.shape[1], -1.)
    x_min = np.full(x.shape[1], -1.)
    for i in range(y.size):
        if y[i] > 5:
            y[i] = 1.
        else:
            y[i] = -1.

        for c in range(x.shape[1]):
            if x_max[c] == -1 or x[i][c] > x_max[c]:
                x_max[c] = x[i][c]
            if x_min[c] == -1 or x[i][c] < x_min[c]:
                x_min[c] = x[i][c]

    for i in range(x.shape[0]):
        x[i] -= x_min
        x[i] /= x_max - x_min

    # Percent of data to use for testing
    test_split = 0.2

    # Split into testing and training sets
    p = np.random.permutation(y.size)
    split_ix = int(y.size * test_split)
    test_p = p[:split_ix]
    train_p = p[split_ix:]

    test_x = x[test_p]
    test_y = y[test_p]
    train_x = x[train_p]
    train_y = y[train_p] 

    # Train classifier
    w = train_classifier(train_x, train_y, 0.008, squared_loss, 0.1, l2_reg, verbose=True)

    # Test classifier
    pred_y = test_classifier(w, test_x)
    score = compute_accuracy(pred_y, test_y)
    print("Final score: %f" % score)

    return None

def tests():
    w = np.array([1., -1., 2., -2.])
    if l1_reg(w) == 6.:
        print("test 1 passed")
    if l2_reg(w) == 10.:
        print("test 2 passed")

    train_y = np.array([1., 1., -1., 1., -1])
    pred_y = np.array([1., -1., -1., 1., 1.])

    if compute_accuracy(train_y, pred_y) == 0.6:
        print("test 3 passed")

    print("Squared loss: %f" % squared_loss(train_y, pred_y))
    print("Hinge loss: %f" % hinge_loss(train_y, pred_y))
    print("Exp loss: %f" % exponential_loss(train_y, pred_y))

if __name__ == "__main__":
    main()
