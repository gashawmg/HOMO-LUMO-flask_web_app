# import the necessary libraries
from flask import Flask, render_template, request
from markupsafe import Markup
#------------------------------------
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from IPython.display import display
from rdkit.Chem import Draw
import mols2grid
#------------------------------------
import pandas as pd
import numpy as np
#------------------------------------
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
#-----------------------------------
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
#----------------------------------
import pickle
import base64
import io
from io import BytesIO


#--------- Import trained lgbm and hgbr regressors 

with open('model.pkl','rb') as f:
         model = pickle.load(f)
with open('scaler.pkl','rb') as f:
          scaler = pickle.load(f)
#--------- we need to use this 106 descriptors for prediction
descriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'MolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'FpDensityMorgan1', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi2v', 'Chi3v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA10', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'NHOHCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'fr_Al_OH', 'fr_Ar_N', 'fr_COO', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_aniline', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_halogen', 'fr_ketone', 'fr_ketone_Topliss', 'fr_methoxy', 'fr_para_hydroxylation']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
   
    csv_file=None
    df1 = None
    df3 =None
    textbox_img = None
    file_imgs =None
    plot = generate_plot()
 
    if request.method == 'POST':
        if 'text_input' in request.form:
            # Process the text input
            data = request.form['text_input']
            if data!="":
                # Put smiles in dataframe
                df = pd.DataFrame(eval(data),columns=["SMILES"])
                #========= function call to calculate 200 molecular descriptors using SMILES
                Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df['SMILES'])
                 #========= Put the 200 molecular descriptors in  dataframe or table
                Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
                #========= Use only the 101 descriptors listed above
                test_dataset_with_106_descriptors = Dataset_with_200_descriptors[descriptors] # these descriptors should be used for predictions
                #======== The data was standardized using standard scaler
                test_scaled = scaler.transform(test_dataset_with_106_descriptors)
                #======== Prediction of HOMO-LUMO energy gap using model1(LightGBMR)
                lgbm_preds = model.predict(test_scaled)
                prediction = np.round(lgbm_preds,2)
                df1 = pd.DataFrame(columns=['SMILES','Predicted'])
                df1['SMILES'] =df['SMILES'].values
                df1['Predicted']= prediction
                
                textbox_img = mols2grid.display(df1,subset=['img', 'Predicted'])
            elif 'csv_file' in request.files:
                # access uploaded file
                uploaded_file = request.files['csv_file']   
                if uploaded_file != None:
                    # Handle the SMILES column
                    df2 = pd.read_csv(uploaded_file)
                    Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df2['SMILES'])
                    Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
                    #========= Use only the 101 descriptors listed above
                    test_dataset_with_106_descriptors = Dataset_with_200_descriptors[descriptors] # these descriptors should be used for predictions  
                    #======== The data was standardized using standard scaler
                    test_scaled = scaler.transform(test_dataset_with_106_descriptors) 
                    #======== Prediction of toxicity using model(LightGBM)
                    lgbm_preds = model.predict(test_scaled)
                    prediction = np.round(lgbm_preds,2)
                    df3 = pd.DataFrame(columns=['SMILES','Predicted'])
                    df3['SMILES'] =df2['SMILES'].values
                    df3['Predicted']= prediction

                    file_imgs =  mols2grid.display(df3,subset=['img','Predicted'])

    return render_template('index.html',textbox_img=textbox_img,file_imgs=file_imgs,plot=plot,df1=df1,df3=df3,csv_file=csv_file,filedownload=filedownload)
def generate_plot():
    # Import test data that contains predicted values to plot actual and predicted dataset
    test = pd.read_csv('actual_vs_predict.csv')

    # -------- Plot the figure of the test dataset on the webpage
    plt.figure(figsize=(8, 6))
    sn.regplot(x=test.Actual , y=test.Predicted,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
    plt.xlabel('Predicted Energy gap', color='blue')
    plt.ylabel('Observed Energy gap', color ='blue')
    plt.title("Test dataset", color='red')
    plt.grid(alpha=0.2)

    # --------R^2 (coefficient of determination) regression score function: 
    R2 =r2_score(test.Actual, test.Predicted)
    R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))

    #------- Model performance using MAE
    MAE = mean_absolute_error(test.Actual, test.Predicted)

    MAE = mpatches.Patch(label="MAE={:04.2f}".format(MAE))
    plt.legend(handles=[R2, MAE])
     # Save the plot to a BytesIO object
    plot_buf = BytesIO()
    plt.savefig(plot_buf, format='png')
    plot_buf.seek(0)

    # Encode the plot as a base64 string
    plot_base64 = base64.b64encode(plot_buf.read()).decode('utf-8')
    return plot_base64
# ================= Get the names of the 200 descriptors from RDKit
def calc_rdkit2d_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    # Append 200 molecular descriptors to each molecule in a list
    Mol_descriptors =[]
    for mol in mols:
        # Calculate all 200 descriptors for each molecule
        mol=Chem.AddHs(mol)
        descriptors = np.array(calc.CalcDescriptors(mol))
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names
  
def filedownload(data,filename):
    if data is not None:
        csv_data = data.to_csv(index=False)
        b64_data = base64.b64encode(csv_data.encode()).decode()
        href_data = f'data:file/csv;base64,{b64_data}'
        return Markup(f'<a href="{href_data}" download="{filename}">Download {filename}</a>')
    else:
        print("the data is missed")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)
