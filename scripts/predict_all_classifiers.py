import numpy as np
import rdkit
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import PandasTools
import pickle
import pharm_map.pharmacophore as ph4
import argparse
import yaml
import gzip
import os
import sklearn as skl

test_supp = gzip.open('EGFR_test_ligands_v2.sdf.gz')
with Chem.ForwardSDMolSupplier(test_supp) as s:
    test_mols = [Chem.AddHs(m) for m in s if m is not None]

mapper = ph4.PharmMapper.from_pickle('EGFR_gaussian_ph4_map.pkl')
mapper.test_mols = test_mols

with open('EGFR_gaussian_classifiers.pkl','rb') as file:
    classifiers = pickle.load(file)

with open('EGFR_params.yml','r') as file:
    params = yaml.safe_load(file)
mapper.scaffold = Chem.RemoveHs(mapper.scaffold)
mapper.scaffold = Chem.AddHs(mapper.scaffold)
_=ph4.align_conformers(mapper.test_mols,ref_mol=mapper.scaffold)
score_mat = mapper.calculate_score_matrix(mapper.test_mols)
scaled_scores = mapper.scaler.fit_transform(score_mat)

labels = ['SVC','SGD','KNeighbors','HGB']
results_dict = {'SMILES':[Chem.MolToSmiles(m) for m in mapper.test_mols],'true_class':[m.GetProp('Class') for m in mapper.test_mols]}
# predict class probabilities with all classifiers
for i in range(len(classifiers)):
    probs = classifiers[i].predict_proba(score_mat)
    inactive_label = labels[i]+'_P(inactive)'
    active_label = labels[i]+'_P(active)'
    predict_label = labels[i]+'_predicted_class'
    results_dict[inactive_label]=probs[:,0]
    results_dict[active_label]=probs[:,1]
    results_dict[predict_label]=[1 if probs[j,1]>=probs[j,0] else 0 for j in range(len(probs)) ]
results = pd.DataFrame(results_dict)
results.to_csv('EGFR_all_classifiers_results.csv')


