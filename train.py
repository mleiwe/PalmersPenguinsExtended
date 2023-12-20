'''This is the training script for the palmers penguin data
This is the train function, this file will read the './Data/palmerpenguins_extended.csv' which contains the extended penguins set. 

Training for a XGBoost classifier undergoes the following steps
1) Drop the `islands` column
2) Encode categorical data (ordinal and one hot encoding)
3) Normalise with the SkLearn MinMaxScaler
4) Evaluate the MLPC (Mulit-layer Perceptron Classifier) on the data set with 5 folds
5) Train the model on the whole data set
6) Save it as a binary file using pickle

Inputs:
Input variables are listed below in the "Input Variables" Section
data_file: file path for the csv
Model_Params: The parameter settings that have been calculated from the grid search in Notebook.ipynb
Model_Filename: The filename of the output file

Output:
Binary file of the model

Requirements
* pandas
* numpy
* scikit-learn

'''

### Libraries Needed 
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

### Input Variables
data_file = "./Data/palmerpenguins_extended.csv"
Model_Params = {
    'activation': 'tanh', 
    'alpha': 0.001, 
    'batch_size': 'auto', 
    'beta_1': 0.9, 
    'beta_2': 0.999, 
    'early_stopping': False, 
    'epsilon': 1e-08, 
    'hidden_layer_sizes': 6, 
    'learning_rate': 'constant', 
    'learning_rate_init': 0.001, 
    'max_fun': 15000, 
    'max_iter': 500, 
    'momentum': 0.9, 
    'n_iter_no_change': 10, 
    'nesterovs_momentum': True, 
    'power_t': 0.5, 
    'random_state': 42, 
    'shuffle': True, 
    'solver': 'lbfgs', 
    'tol': 0.0001, 
    'validation_fraction': 0.1, 
    'verbose': False, 
    'warm_start': False
} 
Model_Filename = 'model.bin'

### Load the whole dataset
print('Reading in training data...')
df = pd.read_csv(data_file)

### Split into variables
X = df.drop(labels=['species','island','year'], axis=1)
y = df['species'].replace(['Adelie','Chinstrap','Gentoo'],[0,1,2])

### Encode the data
def EncodingAndNormalisation(X):
    #Ordinal encoding
    Life_dict = {'chick':0,
                'juvenile':1,
                'adult':2}
    Health_dict = {'underweight':0,
                'healthy':1,
                'overweight':2}
    Sex_dict = {'female':0,
            'male':1}
    EncodeDict = {'sex': Sex_dict,
                'life_stage': Life_dict,
                'health_metrics': Health_dict}
    X = X.replace(EncodeDict)

    #One hot encoding on a single column
    Xi = X.copy().drop(['diet'], axis=1)
    nVals = len(Xi)
    Xi['diet_fish']=np.zeros(nVals)
    Xi['diet_krill']=np.zeros(nVals)
    Xi['diet_parental']=np.zeros(nVals)
    Xi['diet_squid']=np.zeros(nVals)
    for i, (_, row) in enumerate(X.iterrows()):
        d = row['diet']
        if d == 'fish':
            Xi.iloc[i]['diet_fish'] = 1
        
        if d == 'krill':
            Xi.iloc[i]['diet_krill'] = 1
        
        if d == 'parental':
            Xi.iloc[i]['diet_parental'] = 1
        
        if d == 'squid':
            Xi.iloc[i]['diet_squid'] = 1
    X = Xi

    ### Normalise with MinMaxScaler
    mms = MinMaxScaler()
    mms.fit(X)
    X = mms.transform(X)
    return X, mms
X, mms = EncodingAndNormalisation(X)

### Get some stats about the fit by doing K-Folds
print('Evaluating model performance...')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = np.zeros(5)
c = 0
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    
    mlpc = MLPClassifier(**Model_Params)
    mlpc.fit(X_train,y_train)
    y_pred = mlpc.predict(X_test)
    f1 = f1_score(y_test,y_pred, average='micro') #Micro - Calculate metrics globally by counting the total true positives, false negatives and false positives.
    f1_scores[c] = f1
    result = f'Fold Number: {c+1}, F1 score: {np.round(f1,4)}'
    print(result)
    c += 1
result = f'Mean F1 Score: {np.mean(f1_scores)}'
print(result)

## Now train the final model
print('Training final model...')
mlpc = MLPClassifier(**Model_Params)
mlpc.fit(X,y)

print('Saving final model...')
with open(Model_Filename, 'wb') as f_out: # 'wb' means write-binary
    pickle.dump((mms, mlpc, Model_Params), f_out)

print('Model saved.')