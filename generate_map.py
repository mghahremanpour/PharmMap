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

parser = argparse.ArgumentParser()
parser.add_argument('molfile',help='Path to sdf containing rdMol objects to train mapper')
parser.add_argument('params',help='Path to yaml file containing parameters for feature generation')
parser.add_argument('-p','--potency',default='IC50',help="Name of property indicating compound potency")
parser.add_argument('--pt', default=1,help='Cutoff potency value for determining actives vs inactives (compounds at or below this value are active)')
parser.add_argument('-o','--output',default='screen',help='Output prefix')
parser.add_argument('--outdir',default='./',help='Directory to save outputs')
parser.add_argument('-u','--no-unpack',action='store_false',help=
                    'Do not unpack conformers. Only pass this if you know all mols in molfile have exactly one conformer each and you want to keep exactly those conformers')
parser.add_argument('-v','--verbose',action='store_true',help='Print progress messages')

args = parser.parse_args()

# Load params
with open(args.params,'r') as file:
    params = yaml.safe_load(file)
# Make output directory if it doesn't already exist
os.makedirs(args.outdir,exist_ok=True)

# Load training molecules and add explicit Hs (important for alignment, etc.)
if args.verbose:
    print("Loading molecules...",flush=True)
supp = gzip.open(args.molfile)
with Chem.ForwardSDMolSupplier(supp) as s:
    mols = [Chem.AddHs(m) for m in s if m is not None]

# Generate consensus features from training set
mapper = ph4.PharmMapper(mols,feature_factory=params['features']['feature_factory'],
                         potency_key=args.potency)
if args.verbose:
    print("Making consensus pharmacophore features...",flush=True)
mapper.make_consensus_ph4(unpack=args.no_unpack,dist_thresh=params['consensus']['dist_thresh'],
                            potency_thresh=args.pt,
                            method=params['consensus']['clust_method'],
                            max_n_hits=params['consensus']['max_hit_feats'],
                            max_n_decoys=params['consensus']['max_decoy_feats'],
                            sim_cutoff=params['consensus']['sim_cutoff'],
                            dr=params['features']['default_radius'],
                            random_state=params['random_seed'])

# Save consensus mapper
outfile = args.outdir+args.output+'_mapper_featsonly.pkl'
mapper.save(outfile)
if args.verbose:
    print("Done!",flush=True)