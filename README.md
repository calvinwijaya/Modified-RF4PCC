# Modified-RF4PCC
This repository is modified version of [RF4PCC](https://github.com/3DOM-FBK/RF4PCC), utilize Scikit-Learn libraries for supervised point cloud classification.

## Dependencies
- Python3
- Numpy
- Laspy
- Sckikit-learn

## Requirements
Requirements needed is the same with [original repo](https://github.com/3DOM-FBK/RF4PCC), but now the training, test, and validation data is in las format.
1. Training las: a portion of your point cloud with associated geometric and/or radiometric features and a class index (coming after the manual annotation)
2. Validation las: another portion of the point cloud with the same features in the same order and again the manually annotated class index (the classifier will use this file to evaluate the performance of the classification).
3. Test las: the rest of your dataset with the same features, in the same order
4. Feature and class index file (txt): create a two-lines file, the first line is dedicated to the column index of the features that you are using, the second line is for the column which contain the class index.

For example,considering the following distribution of the point cloud columns

| x | y | z | r | g | b | f1 | f2 | f3 | class_index |
|---|---|---|---|---|---|----|----|----|-------------|
| 0 | 1 | 2 | 3 | 4 | 5 |  6 |  7 |  8 |      9      |

if you want to use f1 f2 f3 as features the txt file will be :
Line_1: 6 7 8
Line_2: 9

## How to run
After you have prepared the aforementhioned files, collect them in a folder together with the train.py and classify.py files.

At a command prompt run:

> python train.py feature_path training_path evaluation_path n_core file_to_save_name

This should result in the creation of:

your classifier model .pkl. The name of this file will be related to the number of random trees that performed the best classification (i.e. ne50none.pkl).
a new .las file containig the evaluation dataset with a new column with the predicted classes
To extend the classification to the test dataset at a command prompt run:

> python classify.py feature_path classifier_path test_path file_to_save_name

This should result in the creation of your test file classified (the predicted classes are saved as last column after the features)

## Citation
> Grilli, E.; Remondino, F. Machine Learning Generalisation across Different 3D Architectural Heritage. ISPRS Int. J. Geo-Inf. 2020, 9, 379.

