import time
import pickle
import argparse
import itertools
import numpy as np
import laspy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

feat_to_use = []
class_index = -1
debug = True
x_column_averages = [0.0, 0.0]

def load_features_and_class(filepath):
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]
            elif line_index == 1:
                global class_index
                class_index= int(tokens[0])

def read_data(las_file):
    lasfile = laspy.read(las_file)
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z
    r = lasfile.red
    g = lasfile.green
    b = lasfile.blue
    I = lasfile.intensity

    X_train = np.column_stack((x, y, z, r, g, b, I))
    Y_train = np.array(lasfile.classification)
    
    return X_train, Y_train

def train_model(X_train, Y_train, n_estimators, max_depth, n_jobs):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, oob_score=True, n_jobs=n_jobs)
    model.fit(X_train[:, feat_to_use], Y_train) 
    return model

def write_classification(X, Y, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    las.intensity = X[:, 6]
    las.classification = Y[:]
    las.write(filename + ".las")

def save_model(model, filename):
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description='Train the random forest model.')
    parser.add_argument('features_filepath', help='Path to the file containing the index of the features and the class index')
    parser.add_argument('training_filepath', help='Path to the training file (.txt) [f1, ..., fn, c]')
    parser.add_argument('test_filepath', help='Path to the test file (.txt) [f1, ..., fn, c]')
    parser.add_argument('n_jobs', help='Number of threads used to train the model', type=int)
    parser.add_argument('output_name', help='Name of the predicted test file')
    args = parser.parse_args()
   
    print("Loading data...")
    load_features_and_class(args.features_filepath)

    X_train, Y_train = read_data(args.training_filepath)
    X_test, Y_test = read_data(args.test_filepath)
    
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features with indices: {}'.format(len(Y_train), len(Y_test), feat_to_use))
    ''' ***************************************** TRAINING ************************************** '''
    n_estimators = [50, 100, 150, 200]
    max_depths = [None]
    best_conf = {'ne' : 0, 'md' : 0} 
    best_f1 = 0

    print('\nTraining the model...')  
    start = time.time()                                  
    for ne, md in list(itertools.product(n_estimators, max_depths)):
        model = train_model(X_train, Y_train, ne, md, args.n_jobs)
        Y_test_pred = model.predict(X_test[:, feat_to_use])
        acc = accuracy_score(Y_test, Y_test_pred)
        f1 = f1_score(Y_test, Y_test_pred, average='weighted')
        if f1 > best_f1:
            best_conf['ne'] = ne
            best_conf['md'] = md
            best_f1 = f1
        
        if debug: print('\tne: {}, md: {} - acc: {} f1: {} oob_score: {}'.format(ne, md, acc, f1, model.oob_score_))
    end = time.time()
    print('---> Best parameters: ne: {}, md: {}'.format(best_conf['ne'], best_conf['md']))
    print('---> Feature importance:\n{}'.format(model.feature_importances_))
    print('---> Confusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('---> Training time: {} seconds'.format(end - start))
    ''' ******************************************************************************************** '''

    model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], args.n_jobs)
    Y_test_pred = model.predict(X_test[:, feat_to_use])
    write_classification(X_test, Y_test_pred, args.output_name)
    save_model(model, 'ne{}_md{}.pkl'.format(best_conf['ne'], best_conf['md']))

if __name__== '__main__':
    main()