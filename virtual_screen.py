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


parser = argparse.ArgumentParser()
parser.add_argument('molfile',help='Path to pickle containing rdMol objects to screen')
parser.add_argument('params',help='Path to .yaml file containing parameters for virtual screen')
parser.add_argument('-t','--train',help='Path to pickle containing rdMol objects to train mapper')
parser.add_argument('-m','--mapper',help='Path to pickle with pre-trained mapper information')
parser.add_argument('-p','--potency',default='IC50',help="Name of property indicating compound potency")
parser.add_argument('--pt', default=1,help='Cutoff potency value for determining actives vs inactives (compounds at or below this value are active)')
parser.add_argument('-o','--output',default='screen',help='Output prefix')
parser.add_argument('--outdir',default='./',help='Directory to save outputs')
parser.add_argument('-u','--no-unpack',action='store_false',help=
                    'Do not unpack conformers. Only pass this if you know all mols in molfile have exactly one conformer each')
parser.add_argument('--sim', action='store_true', help='Calculate test-train similarity scores')
parser.add_argument('--sm','--save-mapper', action='store_true', help='Save ph4 mapper information to pickle')
parser.add_argument('--sc','--save-classifiers',action='store_true',help='Save all classifier options rather than just the best one')
parser.add_argument('-v','--verbose',action='store_true',help='Print progress messages')

args = parser.parse_args()
params = yaml.load(args.params)

if args.verbose:
    print("Loading test set molecules...")
with open(args.molfile,'rb') as file:
    tm = pickle.load(file)

if args.train:
    if args.verbose:
        print("Loading training set molecules to train new mapper...")
    with open(args.train,'rb') as file:
        train_mols=pickle.load(file)
    mapper = ph4.PharmMapper(train_mols,test_mols=tm,feature_factory=params['features']['feature_factory'],
                             potency_key=args.p)
    if args.sim:
        if args.verbose:
            print("Calculating test-train similarities...")
        plotfile = args.outdir+args.output+'_test_train_similarities.png'
        sims = mapper.calculate_train_test_similarity(plot=True,plotfile=plotfile)
        simfile = args.outdir+args.output+'_test_train_similarities.csv'
        np.savetxt(simfile,sims,delimiter=',')
    if args.verbose:
        print("Making consensus pharmacophore map...")
    mapper.make_consensus_ph4(unpack=args.no_unpack,dist_thresh=params['consensus']['dist_thresh'],
                              potency_thresh=args.pt,
                              method=params['consensus']['clust_method'],
                              max_n_hits=params['consensus']['max_hit_feats'],
                              max_n_decoys=params['consensus']['max_decoy_feats'],
                              sim_cutoff=params['consensus']['sim_cutoff'],
                              dr=params['features']['default_radius'],
                              random_state=params['random_seed'])
    if args.verbose:
        print("Training optimal classifier...")
    classifier,roc,results=mapper.make_classifier()
elif args.m:
    if args.verbose:
        print("Loading mapper from pickle...")
    mapper = ph4.PharmMapper.from_pickle(args.m)
    mapper.test_mols=tm
    if args.sim:
        if args.verbose:
            print("Calculating test-train similarities...")
        plotfile = args.outdir+args.output+'_test_train_similarities.png'
        sims = mapper.calculate_train_test_similarity(plot=True,plotfile=plotfile)
        simfile = args.outdir+args.output+'_test_train_similarities.csv'
        np.savetxt(simfile,sims,delimiter=',')
else:
    raise ValueError("Need a mapper to screen with - must pass either -t or -m")

if args.verbose:
    print("Predicting test set class probabilities...")
test_probs = mapper.predict()

if args.verbose:
    print("Saving results to .sdf...")
results_df = pd.DataFrame({'Conformer':mapper.test_mols,'P(active)':test_probs[:,1],'P(inactive)':test_probs[:,0]})
results_df['SMILES']=[Chem.MolToSmiles(m) for m in results_df['Conformer']]
results_df.sort_values('P(active)')
outfile = args.outdir+args.output+'_results.sdf'
PandasTools.WriteSDF(results_df,outfile,molColName='Conformer',properties=['SMILES','P(active)','P(inactive)'])

if args.sm:
    if args.verbose:
        print('Saving ph4 mapper...')
    mapfile = args.outdir+args.output+'_ph4_map.pkl'
    mapper.save(mapfile)

# if args.sc:
#     if args.verbose:
#         print("Saving all classifiers...")
#     classifier_file = args.outdir+args.output+'_classifiers.pkl'

if args.verbose:
    print("Done!")