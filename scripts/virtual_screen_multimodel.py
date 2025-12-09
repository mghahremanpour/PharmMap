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

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper

# Make parser to handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('molfile',help='Path to sdf containing rdMol objects to screen')
parser.add_argument('params',help='Path to .yaml file containing parameters for virtual screen')
parser.add_argument('-t','--train',help='Path to sdf containing rdMol objects to train mappers')
parser.add_argument('-m','--mapper',help='Paths to pickles with pre-trained mapper information',nargs='*')
parser.add_argument('-p','--potency',default='IC50',help="Name of property indicating compound potency")
parser.add_argument('--pt', default=1,help='Cutoff potency value for determining actives vs inactives (compounds at or below this value are active)')
parser.add_argument('-o','--output',default='screen',help='Output prefix')
parser.add_argument('--outdir',default='./',help='Directory to save outputs')
parser.add_argument('-u','--no-unpack',action='store_false',help=
                    'Do not unpack conformers. Only pass this if you know all mols in molfile have exactly one conformer each and you want to keep exactly those conformers')
parser.add_argument('--sim', action='store_true', help='Calculate test-train similarity scores')
parser.add_argument('--test-confs',default='all',help='Which conformers from test set to screen ["all" / "best"]')
parser.add_argument('--sm','--save-mapper', action='store_true', help='Save ph4 mapper information to pickle')
parser.add_argument('-c','--confidence',default='confidence.csv',help='Path to file containing confidence information for each test compound ID')
parser.add_argument('--mlp',action='store_true',help='Train multilayer perceptron in addition to other classifier models')
parser.add_argument('-v','--verbose',action='store_true',help='Print progress messages')

# Read and parse command line args
args = parser.parse_args()

# Load params
with open(args.params,'r') as file:
    params = yaml.safe_load(file)
# Make output directory if it doesn't already exist
os.makedirs(args.outdir,exist_ok=True)
# Load test set molecules and add explicit Hs (important for alignment,etc.)
if args.verbose:
    print("Loading test set molecules...",flush=True)
test_supp = gzip.open(args.molfile)
with Chem.ForwardSDMolSupplier(test_supp) as suppl:
    tm = [Chem.AddHs(m) for m in suppl if m is not None]

# train new mappers if pretrained maps weren't provided
if args.train:
    # Load training set molecules and add explicit Hs (important for alignment,etc.)
    if args.verbose:
        print("Loading training set molecules to train new mappers...",flush=True)
    train_supp = gzip.open(args.train)
    with Chem.ForwardSDMolSupplier(train_supp) as suppl:
        train_mols=[Chem.AddHs(m) for m in suppl if m is not None]
    # Get conformers of best active, to use to train mappers
    mol_df = pd.DataFrame({'Molecule':train_mols,'SMILES':[Chem.MolToSmiles(m) for m in train_mols],
                          'Potency':[m.GetDoubleProp(args.potency) for m in train_mols]})
    mol_df.sort_values('Potency',ascending=True,inplace=True)
    grouped = mol_df.groupby('SMILES')
    scaffolds = grouped.get_group(mol_df['SMILES'].iloc[0])
    mappers = []
    count = 0
    for s in scaffolds:
        if args.verbose:
            print(f"Training mapper {count+1} of {len(scaffolds)}")
        mapper = ph4.PharmMapper(train_mols,test_mols=tm,feature_factory=params['features']['feature_factory'],
                             potency_key=args.potency)
        mapper.scaffold = s
        # If desired, calculate maximum similarities between test compounds and training set
        if args.sim and count==1:
            if args.verbose:
                print("Calculating test-train similarities...",flush=True)
            # save histogram and raw values
            plotfile = args.outdir+args.output+'_test_train_similarities.png'
            sims = mapper.calculate_train_test_similarity(plot=True,plotfile=plotfile)
            simfile = args.outdir+args.output+'_test_train_similarities.csv'
            np.savetxt(simfile,sims,delimiter=',')
        # Find consensus pharmacophore features
        if args.verbose:
            print("Making consensus pharmacophore map...",flush=True)
        mapper.make_consensus_ph4(unpack=args.no_unpack,dist_thresh=params['consensus']['dist_thresh'],
                                potency_thresh=args.pt,
                                method=params['consensus']['clust_method'],
                                max_n_hits=params['consensus']['max_hit_feats'],
                                max_n_decoys=params['consensus']['max_decoy_feats'],
                                sim_cutoff=params['consensus']['sim_cutoff'],
                                dr=params['features']['default_radius'],
                                random_state=params['random_seed'],
                                confs='best',
                                verbose=args.verbose) 
        # Train several classifier models and get the best one to use for prediction
        if args.verbose:
            print("Training optimal classifier...",flush=True)
        _=mapper.make_classifier(mlp=args.mlp,verbose=args.verbose)
        mappers.append(mapper)
        count+=1
    
elif args.mapper:
    # Reload pretrained mappers from pickle if provided
    mappers = []
    count = 0
    for f in args.mapper:
        if args.verbose:
            print(f"Loading mapper {count+1} of {len(args.mapper)} from pickle...",flush=True)
        mapper = ph4.PharmMapper.from_pickle(f)
        mapper.test_mols=tm
        if mapper.classifier is None:
            if args.verbose:
                print("Training optimal classifier...",flush=True)
            classifier,roc=mapper.make_classifier(mlp=args.mlp,verbose=args.verbose)
        if args.sim and count==1:
            # If desired, calculate maximum similarities between test compounds and training set
            if args.verbose:
                print("Calculating test-train similarities...",flush=True)
            # Save histogram and raw data
            plotfile = args.outdir+args.output+'_test_train_similarities.png'
            sims = mapper.calculate_train_test_similarity(plot=True,plotfile=plotfile)
            simfile = args.outdir+args.output+'_test_train_similarities.csv'
            np.savetxt(simfile,sims,delimiter=',')
        mappers.append(mapper)
else:
    # Error out if no method to train a mapper was provided
    raise ValueError("Need a mapper to screen with - must pass either -t or -m")

# save mappers
if args.sm:
    if args.verbose:
        print('Saving ph4 mappers...',flush=True)
    count = 0
    for m in mappers:
        mapfile = args.outdir+args.output+'_ph4_map_'+str(count)+'.pkl'
        mapper.save(mapfile)
        count+=1

# Run inactive/active predictions on test set compounds (class 0 is inactive, class 1 is active)
if args.verbose:
    print("Predicting test set class probabilities...",flush=True)
count=0
all_results = []
for mapper in mappers:
    if args.verbose:
        print(f"Doing mapper {count+1} of {len(mappers)}")
    test_probs = mapper.predict(confs=args.test_confs,unpack=args.no_unpack,
                                dist_thresh=params['consensus']['dist_thresh'],verbose=args.verbose)
    results_df = pd.DataFrame({'Conformer':mapper.test_mols,'P(active)':test_probs[:,1],'P(inactive)':test_probs[:,0]})
    results_df['SMILES']=[Chem.MolToSmiles(m) for m in results_df['Conformer']]
    results_df['ID']=[m.GetProp('_Name') for m in results_df['Conformer']]
    results_df['Potency']=[m.GetDoubleProp(args.potency) for m in results_df['Conformer']]
    all_results.append(results_df)
    count+=1
final_df = pd.concat(all_results,ignore_index=True)

# save results
outfile = args.outdir+args.output+'_results.sdf'
PandasTools.WriteSDF(results_df,outfile,molColName='Conformer',properties=['ID','SMILES','Potency','P(active)','P(inactive)'])

if args.verbose:
    print("Done!",flush=True)