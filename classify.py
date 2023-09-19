import time
import pickle
import argparse
import numpy as np
import laspy

feat_to_use = []     # Indices of the features to use. If n is the number of features, from 0 to n-1
x_column_averages = [0.0, 0.0]  # Initialize averages for X columns

def load_features(filepath):
    ''' Load the features indices from a .txt file
       
        Attributes:
            filepath (string)   :  Path to the .txt file
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]

def read_model(filepath):
    ''' Read the Random Forest model from a .pkl file

        Attributes:
            filepath (string)   :   Path to the .pkl file
    '''
    return pickle.load(open(filepath, 'rb'))

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
    
    return X_train

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

def main():
    parser = argparse.ArgumentParser(description='Classify a point cloud with a random forest model.')
    parser.add_argument('features_filepath', help='Path to the file containing the index of the features and the class index')
    parser.add_argument('model', help='Path to .pkl file containing the trained model.')
    parser.add_argument('point_cloud', help='Path to .txt file containing the point cloud to classify.')
    parser.add_argument('output_name', help='Name of the predicted test file')
    args = parser.parse_args()

    start = time.time() 
    print ('Loading data ...')
    load_features(args.features_filepath)
    model = read_model(args.model)
    X = read_data(args.point_cloud)
    
    print ('Classifying the dataset ...')
    
    Y_pred = model.predict(X[:, feat_to_use])

    print ('Saving ...')
    write_classification(X, Y_pred, args.output_name)
    end = time.time()
    print('Data classified in: {}'.format(end - start))

if __name__== '__main__':
    main()