## Libraries required
import pandas as pd
import numpy as np
import pickle

from flask import Flask
from flask import request
from flask import jsonify

## Inputs
model_name = 'model.bin'

with open(model_name, 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    mms, mlpc, MLPC_Params = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])

# Predict functions
def predict():
    penguin = request.get_json() #Converts the JSON into a dictionary

    X = transform_data([penguin])
    penguin_species, y_prob = predict_from_model(X)

    text = f"Type of Penguin: {penguin_species}\nprobability : {y_prob}"
    print(text)
    result = {
        'Species': penguin_species,
        'Probability': float(y_prob)
    }
    return jsonify(result)

## Nested functions (Core logic)
def transform_data(penguin):
    df_X = pd.DataFrame(penguin, index=[0])
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
    df_X = df_X.replace(EncodeDict)

    #One hot encoding
    Xi = df_X.copy().drop(['diet'], axis=1)
    nVals = len(Xi)
    Xi['diet_fish']=np.zeros(nVals)
    Xi['diet_krill']=np.zeros(nVals)
    Xi['diet_parental']=np.zeros(nVals)
    Xi['diet_squid']=np.zeros(nVals)
    for i, (_, row) in enumerate(df_X.iterrows()):
        d = row['diet']
        if d == 'fish':
            Xi.iloc[i]['diet_fish'] = 1
        if d == 'krill':
            Xi.iloc[i]['diet_krill'] = 1
        if d == 'parental':
            Xi.iloc[i]['diet_parental'] = 1
        if d == 'squid':
            Xi.iloc[i]['diet_squid'] = 1
    df_X = Xi.copy()


    ### Min Max Scaler
    X = mms.transform(df_X)

    return X

def predict_from_model(X):
    y_probs = mlpc.predict_proba(X)
    Chosen = np.argmax(y_probs)
    prob = np.max(y_probs)
    if Chosen == 0:
        Species = "Adelie"
    elif Chosen == 1:
        Species = "Chinstrap"
    else:
        Species = "Gentoo"
    return Species, prob

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)