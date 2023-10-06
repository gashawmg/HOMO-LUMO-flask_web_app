from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
#----------------------------------------------------
import pandas as pd
import numpy as np
#----------------------------------------------------
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
#----------------------------------------------------
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
#----------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

# read csv file
dataset = pd.read_csv('D:\Orbital_Energies_input_data.csv')

# generate canonical SMILES
def canonical_SMILES(smiles):
    canon_smi = [Chem.CanonSmiles(smi) for smi in smiles]
    return canon_smi

# Canonical SMILES
Canon_SMILES = canonical_SMILES(dataset.SMILES)

# Put the smiles in the dataframe
dataset['SMILES'] = Canon_SMILES

# Create a list for duplicate smiles
duplicates_smiles = dataset[dataset['SMILES'].duplicated()]['SMILES'].values

# Create a list for duplicate smiles
dataset[dataset['SMILES'].isin(duplicates_smiles)].sort_values(by=['SMILES'])

# 2.  Drop duplicate values
dataset_new = dataset.drop_duplicates(subset=['SMILES'])

#  General molecular descriptors-about 200 molecular descriptors
def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] 
                                    for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 

# Function call
Mol_descriptors,desc_names = RDkit_descriptors(dataset_new['SMILES'])

# put them in dataframe
df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

def remove_correlated_features(descriptors):
    # Calculate correlation
    correlated_matrix = descriptors.corr().abs()

    # Upper triangle of correlation matrix
    upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape),k=1).astype(np.bool_))

    # Identify columns that have above 0.9 values of correlation
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.9)]
    print(to_drop)
    descriptors_correlated_dropped = descriptors.drop(columns=to_drop, axis=1)
    return descriptors_correlated_dropped 
# function call
descriptors_new = remove_correlated_features(df_with_200_descriptors)

#credit to dataprofessor
# adapted from:https://github.com/dataprofessor/padel/blob/main/padelpy.ipynb

from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(descriptors_new, threshold=0.1)
print("number of rows and features:",X.shape)
print(X.columns.values)
# Assign Features and Labels
X  # molecular features
y = dataset_new.Energygap # labels or targets

## 3.  Train-test data split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from lightgbm import LGBMRegressor

# Define LGBMRegressor model
lgbm_opt =LGBMRegressor(max_depth=7,min_samples_split=4, n_estimators= 1000,max_features='auto',random_state=42)

# Train the model and predict the yield using X_test 
lgbm_opt.fit(X_train, y_train)
lgbm_predict = lgbm_opt.predict(X_test)

# Model performance using RMSE
MAE = mean_absolute_error(y_test, lgbm_predict)
print('MAE:', MAE)

# R^2 (coefficient of determination) regression score function: 
R2 =r2_score(y_test, lgbm_predict)
print('R^2:', R2)

sn.regplot(x=lgbm_predict , y=y_test,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted Energy gap', color='blue')
plt.ylabel('Observed Energy gap', color ='blue')
plt.title("Test set", color='red')
plt.grid(alpha=0.2)
R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))
MAE = mpatches.Patch(label="MAE={:04.2f}".format(MAE))
plt.legend(handles=[R2, MAE])
plt.show()

# create a CSV file and save the actual and predicted values of the test data
df_ = pd.DataFrame()
df_['Actual'] = np.array(y_test)
df_['Predicted'] = lgbm_predict
df_.to_csv(r'D:\HOMO_LUMO_flask\actual_vs_predict.csv',index=False)

import pickle
with open('model.pkl','wb') as f:
          pickle.dump(lgbm_opt,f)

with open('scaler.pkl','wb') as f:
          pickle.dump(scaler,f)