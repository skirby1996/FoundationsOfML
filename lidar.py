import numpy as np
from hw3 import *

print(get_id())

def main():
    # Parse data from csv
    data = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('data/test.csv', delimiter=',', skip_header=1)
    x = data[:, 1:-1]
    y = data[:, -1]
    
    test_features = test_data[:, 1:]

    # Convert 0/1 to -1/1 values
    y = (y * 2) - 1
    
    # Normalize features
    x_mean = np.mean(x, axis=0)
    x_stddev = np.std(x, axis=0)

    x = np.subtract(x, x_mean)
    x = np.divide(x, x_stddev)

    test_features = np.subtract(test_features, x_mean)
    test_features = np.divide(test_features, x_stddev)

    # Add column of 1s to learn biases
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    test_features = np.concatenate((np.ones((test_features.shape[0], 1)), test_features), axis=1)

    # Split into testing and training sets
    test_split = 0.2
    p = np.random.permutation(y.size)
    split_ix = int(y.size * test_split)
    test_p = p[:split_ix]
    train_p = p[split_ix:]

    test_x = x[test_p]
    test_y = y[test_p]
    train_x = x[train_p]
    train_y = y[train_p]

    w = train_classifier(train_x, train_y, 0.002, logistic_loss, 0.1, None, 1000, verbose=True)
    pred_y = test_classifier(w, test_x)
    score = compute_accuracy(pred_y, test_y)
    print("\nAcc: %.3f\n" % score)

    # Create CSV submission
    f = open("submission.csv", 'w')
    f.write("ID,IsSinkhole\n")

    pred = test_classifier(w, test_features)
    for i in range(pred.size):
        f.write("%d,%f\n" % (i, pred[i]))

    f.close()

if __name__ == "__main__":
    main()
