import numpy as np
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.mixture import GaussianMixture


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.metrics import binary_accuracy
from keras import initializers

from hw3 import *
    

def score_classifier(pred_y, test_y, split):
    confusion_matrix = np.zeros(shape=(2, 2))
    score = 0
    for ix in range(pred_y.size):
        if pred_y[ix] < split:
            if test_y[ix] < split:
                # True negative
                confusion_matrix[0, 0] += 1
                score += 1
            else:
                # False negative
                confusion_matrix[0, 1] += 1
        else:
            if test_y[ix] < split:
                # False positive
                confusion_matrix[1, 0] += 1
            else:
                # True positive
                confusion_matrix[1, 1] += 1
                score += 1
    return confusion_matrix, score/pred_y.size, average_precision_score(test_y, pred_y)


def main():
    # Parse data from csv
    data = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('data/test.csv', delimiter=',', skip_header=1) 
    
    # Create new arrays of positive/negative exmaples
    data_n = np.zeros(shape=(int(data.shape[0] - np.sum(data[:, -1])), data.shape[1]))
    data_p = np.zeros(shape=(int(np.sum(data[:, -1])), data.shape[1]))
        
    n_weight = data_n.shape[0] / data.shape[0]
    p_weight = data_p.shape[0] / data.shape[0] 
    
    n_ix = 0
    p_ix = 0
    for ix in range(data.shape[0]):
        if data[ix, -1] == 0:
            data_n[n_ix] = data[ix]
            n_ix += 1
        else:
            data_p[p_ix] = data[ix]
            p_ix += 1

    # Undersample negative dataset so training data has an equal
    # number of postive and negative examples
    p = np.random.permutation(data_n.shape[0])
    p = p[:data_p.shape[0]]

    data_n = data_n[p]
    data_b = np.concatenate((data_p, data_n))

    # Shuffle new dataset
    p = np.random.permutation(data_b.shape[0])
    data_b = data_b[p]

    x_b = data_b[:, 1:-1]
    y_b = data_b[:, -1]

    cols = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

    x = data[:, cols]
    y = data[:, -1]

    test_features = test_data[:, cols]

    split = 0.5
    '''
    # Convert 0/1 to -1/1 values
    y = (y * 2) - 1
    y_b = (y_b * 2) - 1
    split = 0
    #'''

    # Normalize features
    x_mean = np.mean(x, axis=0)
    x_stddev = np.std(x, axis=0)

    # Normalized balance
    #x_b = np.subtract(x_b, x_mean)
    #x_b = np.divide(x_b, x_stddev)

    # Normalize full dataset
    x = np.subtract(x, x_mean)
    x = np.divide(x, x_stddev)

    # Normalize testing set
    test_features = np.subtract(test_features, x_mean)
    test_features = np.divide(test_features, x_stddev)

    '''
    # Add column of 1s to learn biases
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    test_features = np.concatenate((np.ones((test_features.shape[0], 1)), test_features), axis=1)
    '''

    # Split into testing and training sets
    '''
    test_split = 0.2
    p = np.random.permutation(y_b.size)
    split_ix = int(y.size * test_split)
    test_p = p[:split_ix]
    train_p = p[split_ix:]
    test_x_b = x_b[test_p]
    test_y_b = y_b[test_p]
    train_x_b = x_b[train_p]
    train_y_b = y_b[train_p]
    '''

    test_split = 0.2
    p = np.random.permutation(y.size)
    split_ix = int(y.size * test_split)
    test_p = p[:split_ix]
    train_p = p[split_ix:]
    test_x = x[test_p]
    test_y = y[test_p]
    train_x = x[train_p]
    train_y = y[train_p]

    # Single neuron
    '''
    w = train_classifier(train_x, train_y, 0.002, logistic_loss, 0.1, None, 1000, verbose=True)
    pred_y = test_classifier(w, test_x)
    score = compute_accuracy(pred_y, test_y)
    print("\nAcc: %.3f\n" % score)
    '''

    # SVM
    ''' 
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']

    for k in kernels:
        print("Kernel: ", k) 
        clf = svm.SVC(
            gamma='scale',
            kernel=k,
            probability=True,
            class_weight={1: 2},
            verbose=False)
        clf.fit(train_x, train_y)
        
        print("Scoring test dataset") 
        pred_y = clf.predict(test_x)
        confusion_matrix, score, m_ap = score_classifier(pred_y, test_y, split)
        print("Score: ", score)
        print("MAP: ", m_ap)
        print(confusion_matrix)
        
        print("Scoring full dataset")
        pred_y = clf.predict(x)
        confusion_matrix, score, m_ap = score_classifier(pred_y, y, split)
        print("Score: ", score)
        print("MAP: ", m_ap)
        print(confusion_matrix)
    #'''
    
    # Random Forest
    '''
    estimators = [100, 150, 200, 250, 300]
    max_depth = [None, 10, 25]

    for e in estimators: 
        for d in max_depth:
            print("Num estimators: ", e)
            print("Max depth: ", -1 if d is None else d)
            clf = RandomForestClassifier(
                    n_estimators=e, 
                    max_depth=d,
                    class_weight={1: n_weight/p_weight})
            clf = clf.fit(train_x, train_y)
             
            print("Scoring on testing dataset")
            pred_y = clf.predict(test_x)
            confusion_matrix, score, m_ap = score_classifier(pred_y, test_y, split)
            print("Score: ", score) 
            print("MAP: ", m_ap)
            print(confusion_matrix)
            
            """
            print("Scoring on full dataset")
            pred_y = clf.predict(x)
            confusion_matrix, score, m_ap = score_classifier(pred_y, y, split)
            print("Score: ", score)
            print("MAP: ", m_ap)
            print(confusion_matrix)
            print("\n")
            #"""
    #'''
    
    # Gaussian Mixture model
    '''
    clf = GaussianMixture(n_components=2, max_iter=10000)
    clf.fit(train_x, train_y)

    print("Scoring on testing set:")
    print("Predict labels: ")
    pred_y = clf.predict(test_x)
    confusion_matrix, score, m_ap = score_classifier(pred_y, test_y, split)
    print("Score: ", score)
    print("MAP: ", m_ap)
    print(confusion_matrix)
    
    print("Predict probs: ")
    pred_y_probs = clf.predict_proba(test_x)
    for i in range(pred_y.size):
        if pred_y_probs[i, 0] >= pred_y_probs[i, 1]:
            pred_y[i] = 0
        else:
            pred_y[i] = 1
    confusion_matrix, score, m_ap = score_classifier(pred_y, test_y, split)
    print("Score: ", score)
    print("MAP: ", m_ap)
    print(confusion_matrix)
    
    #'''

    # Neural Net
    #'''
    # Create model
    model = Sequential()
    model.add(Dense(16,
        bias_initializer=initializers.Constant(value=0.1),
        input_shape=(cols.size,),
        activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(16,
        bias_initializer=initializers.Constant(value=0.1),
        activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(12,
        bias_initializer=initializers.Constant(value=0.1),
        activation='relu'))
    #model.add(BatchNormalization()) 
    model.add(Dense(8,
        bias_initializer=initializers.Constant(value=0.1),
        activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(1,
        #bias_initializer=initializers.Constant(value=0.1),
        activation='sigmoid'))

    model.compile(loss='mse',
                  optimizer='adam')

    # Split train_x into positive/negative sets
    #num_pos = np.sum((train_y + 1) / 2)
    num_pos = np.sum(train_y)
    train_x_n = np.zeros(shape=(int(train_y.size - num_pos), train_x.shape[1]))
    train_y_n = np.full(shape=(train_x_n.shape[0],), fill_value=0.)
    train_x_p = np.zeros(shape=(int(num_pos), train_x.shape[1]))
    train_y_p = np.full(shape=(train_x_p.shape[0],), fill_value=1.)

    n_ix = 0
    p_ix = 0
    for ix in range(train_y.size):
        if train_y[ix] < split:
            train_x_n[n_ix] = train_x[ix]
            n_ix += 1
        else:
            train_x_p[p_ix] = train_x[ix]
            p_ix += 1

    # Undersample negative dataset so training data has an equal
    # number of postive and negative examples
    p = np.random.permutation(train_y_n.size)
    p = p[:train_y_p.size]

    sample_x_n = train_x_n[p]
    sample_y_n = train_y_n[p]
    
    rounds = 15
    epochs_per_round = [100, 100, 100, 150, 150, 150, 200, 200, 200, 250, 250, 250, 250, 250, 250]
    for r in range(rounds):
        print("Round %d/%d" % (r + 1, rounds))
        train_x_b = np.concatenate((train_x_p, sample_x_n))
        train_y_b = np.concatenate((train_y_p, sample_y_n))

        # Shuffle new dataset
        p = np.random.permutation(train_y_b.size)
        train_x_b = train_x_b[p]
        train_y_b = train_y_b[p]

        # Fit on dataset
        model.fit(train_x_b, train_y_b, epochs=100, batch_size=train_y.size, verbose=0) 
        
        # Show performance
        _, score, m_ap = score_classifier(model.predict(train_x), train_y, split)
        print("Score: %f\nMAP: %f" % (score, m_ap))

        # Reclassify negatives
        pred_y_n = model.predict(train_x_n)

        # Find hard negatives 
        scored_y_n = np.concatenate((train_x_n, pred_y_n), axis=1)
        n_ix = np.argsort(-scored_y_n[:, -1])  
        
        # Oversample hardest negatives, sample from randomly
        p_factor = 1
        n_ix = n_ix[:int(train_y_p.size * p_factor)]
        p = np.random.permutation(n_ix.size)
        n_ix = n_ix[p]
        sample_x_n = scored_y_n[n_ix[:train_y_p.size], :-1] 

    # Score model
    print("Validating using testing dataset")
    pred_y = model.predict(test_x, batch_size=test_y.size) 
       
    confusion_matrix, score, m_ap = score_classifier(pred_y, test_y, split)
    print("Score: ", score)
    print("MAP: ", m_ap)
    print(confusion_matrix)

    print("Validating using full dataset")
    pred_y = model.predict(x, batch_size=y.size)
    
   
    confusion_matrix, score, m_ap = score_classifier(pred_y, y, split)
    print("Score: ", score)
    print("MAP: ", m_ap)
    print(confusion_matrix)
    #'''

    # Generate submission
    #'''
    # Create CSV submission
    f = open("submission.csv", 'w')
    f.write("ID,IsSinkhole\n")

    #pred = test_classifier(w, test_features)
    pred = model.predict(test_features, batch_size=test_y.size)
    #pred = clf.predict(test_features)
    
    for i in range(pred.size):
        f.write("%d,%f\n" % (i, pred[i]))

    f.close()
    #'''

if __name__ == "__main__":
    main()
