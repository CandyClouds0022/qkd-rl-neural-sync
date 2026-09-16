"""
Training and testing steps for the BASELINE model (dummy) of QKD protocol:
The file original aim was to train and test a model to produce more samples, expanding the original
dataset, generated ny the physicalengine file.
Reminder: it's a static dataset (sweep on distances L, fixed bias@0.5, noise), so the ai agent has a 
conscribed scenario to train onto (betterment is supposed to be LIMITED).
Initial idea: K-fold cross validation (avoids overfitting) on a massive dataset.

NOTEForUser scaling NEEDED, else the MLNP risks considering distances (>> in magnitude) as of 
greater importance  wrt QBER (0-1)-> scaling helps regulating the weghts of the NN

reminder: R2 (0-1, perfect abt 0.9) is the coefficient that indicates how well the model follows the 
actual data; MAE reflects the average error in the predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import time

#OLD version: BT locals
base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir, "dummy_model_tpm")
os.makedirs(output_dir, exist_ok=True) #creating folder if not existing already
input_csv_path=os.path.join(output_dir, "dataset_dummy_fortraintest.csv") 
df=pd.read_csv(input_csv_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#features
X=df[['distance']].values
#target
Y=df[['QBER', 'key_length']].values

kf=KFold(n_splits=5, shuffle=True, random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()

fold=1
for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_test=X[train_index], X[test_index]
    Y_train, Y_test=Y[train_index], Y[test_index]
    
    #scaling
    X_train_scaled=scaler_x.fit_transform(X_train)
    Y_train_scaled=scaler_y.fit_transform(Y_train)
    X_test_scaled=scaler_x.transform(X_test)
    Y_test_scaled=scaler_y.transform(Y_test)
    
    #model
    model=MLPRegressor(hidden_layer_sizes=(64, 32), solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, Y_train)
    
    #predictions
    predictions=model.predict(X_test_scaled)
    Y_pred=scaler_y.inverse_transform(predictions)
    mae=mean_absolute_error(Y_test, predictions)
    r2=r2_score(Y_test, predictions)
    
    print(f"Fold {fold} - MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    fold+=1

# saving stepppp => final model trained on the whole dataset (after k-fold validation)
X_scaled=scaler_x.fit_transform(X)
model.fit(X_scaled, Y)

# evaluations on times and performance statustìcs
start_time=time.perf_counter()
final_predictions=model.predict(X_scaled)
end_time=time.perf_counter()

inference_time_total=end_time - start_time
avg_time_per_sample=inference_time_total / len(X)

# adding some cols to DataFrame
df['pred_QBER']=final_predictions[:, 0]
df['pred_key_length']=final_predictions[:, 1]
df['abs_err_QBER']=np.abs(df['QBER']-df['pred_QBER']) #suggested :)
#aka: MAE==medie and means LIE 
df['abs_err_key_length']=np.abs(df['key_length']-df['pred_key_length'])

# savings (using the nw paths)
model_path=os.path.join(output_dir, "static_ai_model.pkl")
scaler_path=os.path.join(output_dir, "scaler_x.pkl")
csv_results_path=os.path.join(output_dir, "dataset_with_results.csv")

joblib.dump(model, model_path)
joblib.dump(scaler_x, scaler_path)

# df.to_csv(csv_results_path, index=False)
csv_results_path=os.path.join(output_dir, "dataset_with_results_tpm.csv")
df.to_csv(csv_results_path, index=False)

print(f"Inference total time: {inference_time_total:.6f}s")
print(f"Average time per sample: {avg_time_per_sample:.6f}s")
print(f"Model and results saved in: {output_dir}")