import numpy as np
import rdkit
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDConfig, Geometry, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures, rdDistGeom, rdMolTransforms, rdShapeAlign, FeatMaps, Draw
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Numerics import rdAlignment
from rdkit.Chem.Draw import IPythonConsole, MolDrawing
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import os
import itertools
import copy
import pickle
import sklearn as skl
import umap
DrawingOptions.includeAtomNumbers=True

def optimal_kmeans(data,min_k=2,max_k=10,random_state=None):
    min_silhouette = -1
    silhouettes = []
    best_k=min_k
    for k in range(min_k,max_k):
        clusterer = skl.cluster.KMeans(n_clusters=k,random_state=random_state)
        clust_labels = clusterer.fit_predict(data)
        silhouette_avg = skl.metrics.silhouette_score(data,clust_labels)
        silhouettes.append(silhouette_avg)
        if silhouette_avg>min_silhouette:
            best_k = k
            min_silhouette = silhouette_avg
    opt_clusterer = skl.cluster.KMeans(n_clusters=best_k,random_state=random_state)
    opt_clusts = opt_clusterer.fit_predict(data)
    return opt_clusts, best_k, min_silhouette, silhouettes

def optimal_hierarchical_clustering(data,min_nclust,max_nclust):
    min_silhouette = -1
    silhouettes = []
    best_n=min_nclust
    for n in range(min_nclust,max_nclust):
        clusterer = skl.cluster.AgglomerativeClustering(n_clusters=n)
        clust_labels = clusterer.fit_predict(data)
        silhouette_avg = skl.metrics.silhouette_score(data,clust_labels)
        silhouettes.append(silhouette_avg)
        if silhouette_avg>min_silhouette:
            best_n = n
            min_silhouette = silhouette_avg
    opt_clusterer = skl.cluster.AgglomerativeClustering(n_clusters=best_n)
    opt_clusts = opt_clusterer.fit_predict(data)
    return opt_clusts, best_n, min_silhouette, silhouettes

def generate_LE_conformers(mols,embed_count=100,random_seed=-1,save=False,filename=None):
    out_mols = []
    for mol in mols:
        confs = AllChem.EmbedMultipleConfs(mol,numConfs=embed_count,
                                            randomSeed=random_seed)
        res=AllChem.MMFFOptimizeMoleculeConfs(mol)
        LE_conf_ID = res.index(min(res,key=lambda t: t[1]))
        out_mol = Chem.Mol(mol,confId=LE_conf_ID)
        out_mols.append(out_mol)
    if save:
        if filename:
            save_file = filename
        else:
            save_file = "LE_conformers.pkl"
        with open(save_file,"wb") as file:
            pickle.dump(out_mols,file)
    return out_mols
                
def get_representative_conformers(mol,thresh=1.5,make_mols=False):
    '''
    Generate set of representative conformers from an input molecule
    Args:
    mol: molecule (with conformers already generated) to make conf set from
    '''
    dist_matrix = AllChem.GetConformerRMSMatrix(mol)
    cids = [c.GetId() for c in mol.GetConformers()]
    clusts = Butina.ClusterData(dist_matrix,len(cids),distThresh=thresh,isDistData=True,reordering=True)
    centroids = [x[0] for x in clusts]
    centroid_cids = [cids[i] for i in centroids]
    if make_mols:
        new_mols=[]
        for c in centroid_cids:
            nm = Chem.Mol(mol,confId=c)
            new_mols.append(nm)
    else:
        new_mols=None
    return clusts, centroid_cids, new_mols

def extract_features(mols,feature_factory=None):
    if feature_factory == None:
        feature_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    families = []
    xs = []
    ys = []
    zs = []
    for m in mols:
        feats = feature_factory.GetFeaturesForMol(m)
        for f in feats:
            if f.GetFamily() != "LumpedHydrophobe": # temporary kludge to work around RDKit's bad feature definitions here
                families.append(f.GetFamily())
                xs.append(f.GetPos().x)
                ys.append(f.GetPos().y)
                zs.append(f.GetPos().z)
    feat_df = pd.DataFrame({"Family":families,'x':xs,'y':ys,'z':zs})
    return feat_df

def align_conformers(mols,ref_mol=None):
    '''
    Canonicalize all conformers of input molecules and align them
    Args:
    mols: iterable of rdMol objects
    Returns:
    '''
    if ref_mol==None:
        ref_mol = mols[0]
    ref_params = AllChem.MMFFGetMoleculeProperties(ref_mol)
    RMSDs = []
    scores = []
    for m in mols:
        rdMolTransforms.CanonicalizeMol(m)
        mmff_params=AllChem.MMFFGetMoleculeProperties(m)
        mol_scores={}
        mol_RMSDs = {}
        aligners = rdMolAlign.GetO3AForProbeConfs(m,ref_mol,1,mmff_params,ref_params)
        for i in range(len(aligners)):
            rmsd = aligners[i].Align()
            conf = m.GetConformers()[i]
            cid=conf.GetId()
            conf_score = aligners[i].Score()
            mol_scores[cid]=conf_score
            mol_RMSDs[cid]=rmsd
        scores.append(mol_scores)
        RMSDs.append(mol_RMSDs)
    return RMSDs,scores

def cluster_features(feats,clust_method='hierarchical',max_n=10,random_state=None):
    clustered_feats = []
    clust_qc = {}
    feats_by_fam = feats.groupby('Family')
    for fam, group in feats_by_fam:
        pos_matrix = group[['x','y','z']].to_numpy()
        if clust_method == 'hierarchical':
            opt_clusts, best_n, min_silhouette, silhouettes = optimal_hierarchical_clustering(pos_matrix,
                                                                                              min_nclust=2,
                                                                                              max_nclust=max_n)
        elif clust_method=='k_means':
            opt_clusts,best_n,min_silhouette,silhouettes = optimal_kmeans(pos_matrix, min_k=2, max_k=max_n,
                                                                          random_state=random_state)
        group['Cluster'] = opt_clusts
        if len(clustered_feats)==0:
            clustered_feats = [group]
        else:
            clustered_feats.append(group)
        clust_qc[fam] = (best_n,min_silhouette,silhouettes)
    clustered_feats = pd.concat(clustered_feats).reset_index(drop=True)
    return clustered_feats, clust_qc

def compute_feature_centroid_sigma(feats):
    consensus_feats = []
    feats_by_fam = feats.groupby('Family')
    for fam, group in feats_by_fam:
        clusts = group.groupby('Cluster')
        for clust, g in clusts:
            n_feats = len(g)
            if n_feats > 1:
                pos_matrix = g[['x','y','z']].to_numpy()
                centroid = np.average(pos_matrix,axis=0)
                cov = np.cov(pos_matrix.T)
                sigma = np.average(np.diag(cov))
            else:
                centroid = g[['x','y','z']].to_numpy()
                sigma = 1.0265
            consensus_clust = pd.DataFrame({'Family':[fam],'x':[centroid[0]],'y':[centroid[1]],
                                            'z':[centroid[2]],'sigma':[sigma],'n_feats':[n_feats]})
            consensus_feats.append(consensus_clust)
    consensus_feats = pd.concat(consensus_feats).reset_index(drop=True)
    return consensus_feats

def cluster_by_fingerprint(mols,fp_method='RDKit',all_confs=False,
                        clust_method='k_means',max_k=10,random_state=None):
    if fp_method == 'RDKit':
        fpgen = AllChem.GetRDKitFPGenerator()
    elif fp_method == 'atom_pair':
        fpgen = AllChem.GetAtomPairGenerator()
    elif fp_method == 'torsion':
        fpgen = AllChem.GetTopologicalTorsionGenerator()
    elif fp_method == 'morgan':
        fpgen = AllChem.GetMorganGenerator()
    else:
        print("Unsupported fingerprint generation method. Please use RDKit, atom_pair, torsion, or morgan")
        return None
    fps = [fpgen.GetFingerprint(m) for m in mols]
    fp_array = np.asarray(fps)
    
    # dimensionality reduction
    reducer = umap.UMAP(metric='jaccard',random_state=random_state)
    umap_fit = reducer.fit_transform(fp_array)
    
    # clustering
    if clust_method == 'k_means':
        clusts, best_k, min_silhouette, silhouettes = optimal_kmeans(umap_fit,max_k=max_k,
                                                                    random_state=random_state)
        clust_qc = (best_k,min_silhouette,silhouettes)
    
    # make dict of molecules by cluster
    clust_dict={}
    largest_size=0
    largest_clust = -1
    for c in list(set(clusts)):
        clust_mols = [mols[i] for i in range(len(mols)) if clusts[i]==c]
        clust_dict[int(c)] = clust_mols
        if len(clust_mols)>largest_size:
            largest_size=len(clust_mols)
            largest_clust = int(c)
    return clust_dict,umap_fit,clusts,largest_clust,clust_qc

def get_pharmacophore_alignment(query,ref,verbose=False):
    '''
    Align two pharmacophores using Kabsch algorithm
    Args:
    query: rdPharm3D object to be aligned
    ref: rdPharm3D object to align against
        ref should contain only essential ph4 features
    verbose: whether to print diagnostic messages
    Returns:
    opt_feat_ids: list of indices for features in query that best
        match ref
    opt_mat: transform matrix to rotate query to optimally align to ref
    opt_rssd: root sum squared distance between query and ref, after
        alignment
    results: pd.DataFrame with all feature combinations, transform
        matrices, and rssd values'''
    # extract features from rdPharm3D objects
    ref_feats = ref.getFeatures()
    ref_df = pd.DataFrame({'family':[f.GetFamily() for f in ref_feats],
                        'x':[list(f.GetPos())[0] for f in ref_feats],
                        'y':[list(f.GetPos())[1] for f in ref_feats],
                        'z':[list(f.GetPos())[2] for f in ref_feats]})
    query_feats = query.getFeatures()
    query_df = pd.DataFrame({'family':[f.GetFamily() for f in query_feats],
                        'x':[list(f.GetPos())[0] for f in query_feats],
                        'y':[list(f.GetPos())[1] for f in query_feats],
                        'z':[list(f.GetPos())[2] for f in query_feats]})
    
    # get possible combinations of query features that match reference
    match_found = [False]*len(ref_df)
    matches = [None]*len(ref_df)
    for i in range(len(ref_df)):
        fam = ref_df['family'].iloc[i]
        fmatch = list(query_df.index[query_df['family']==fam])
        if len(fmatch)>0:
            matches[i]=fmatch
            match_found[i]=True
    if sum(match_found)<len(ref_df) and verbose:
        print("Warning: Not all reference features have possible matches in query")
    # remove unmatched features from reference
    # (they won't be useful for upcoming calculations)
    ref_matched = ref_df.iloc[match_found]
    ref_pos = ref_matched[['x','y','z']].to_numpy()
    
    # center reference features (subtract centroid from position)
    ref_centroid = np.mean(ref_pos,axis=0)
#     ref_pos = ref_pos-ref_centroid
    
    # test all combinations of query feature matches
    summary_feats = []
    summary_mats = []
    summary_rssd = []
    matches = [m for m in matches if m is not None]
    for q in list(itertools.product(*matches)):
        # skip any combinations that double-count a feature
        if len(np.unique(q))<len(q):
            continue
        # get transform matrix; try both unchanged and reflected probe configurations
        pos = query_df[['x','y','z']].iloc[list(q)].to_numpy()
        centroid = np.mean(pos,axis=0)
        pos = pos-centroid+ref_centroid
        SSD, mat = rdAlignment.GetAlignmentTransform(ref_pos,pos)
        SSD_reflect, mat_reflect = rdAlignment.GetAlignmentTransform(ref_pos,pos,reflect=True)
        summary_feats.append(list(q))
        if SSD_reflect<SSD:
            summary_mats.append(mat_reflect)
            summary_rssd.append(SSD_reflect)
        else:
            summary_mats.append(mat)
            summary_rssd.append(SSD)
        # compute optimal rotation
#         rot, rssd = R.align_vectors(ref_pos,pos)
#         rmat = rot.as_matrix()
#         summary_feats.append(list(q))
#         summary_mats.append(rmat)
#         summary_rssd.append(rssd)
    results = pd.DataFrame({'Feature IDs':summary_feats,
                            'Transform Matrix':summary_mats,
                            'RSSD':summary_rssd})
    # get optimal feature combination and corresponding transform/rssd
    opt_id = results['RSSD'].idxmin()
    opt_feat_ids,opt_mat,opt_rssd = results.loc[opt_id].tolist()
    
    return opt_feat_ids,opt_mat,opt_rssd,results

class PharmMapper:
    ff_path = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    ff=ChemicalFeatures.BuildFeatureFactory(ff_path)
    hits=[]
    decoys=[]

    def __init__(self,mols,feature_factory=None,potency_key='IC50'):
        self.mols=mols
        if feature_factory:
            self.ff = feature_factory
        self.potkey = potency_key
    
    def __unpack_conformers(self,rep_only=True,dist_thresh=1.5):
        new_mols=[]
        for m in self.mols:
            if len(m.GetConformers())>1:
                if rep_only:
                    _,_,mols_to_use = get_representative_conformers(m,thresh=dist_thresh,
                                                                    make_mols=True)
                    new_mols = new_mols + mols_to_use
                else:
                    for c in m.GetConformers():
                        nm=Chem.Mol(m,confId=c.GetId())
                        new_mols.append(nm)
            else:
                new_mols.append(m)
        return new_mols

    def __split_actives_inactives(self,thresh=0.1):
        for m in self.training_mols():
            if m.GetDoubleProp<=thresh:
                self.hits.append(m)
            else:
                self.decoys.append(m)

    def prepare_training_set(self,dist_thresh=1.5,potency_thresh=0.1):
            self.training_mols = self.__unpack_conformers(rep_only=True,dist_thresh=dist_thresh)
            RMSDs,scores = align_conformers(self.training_mols)
            self.__split_actives_inactives(thresh=potency_thresh)


    def generate_training_features(self,clust_method='hierarchical',max_n=10,random_state=None):
        allhits = extract_features(self.hits,self.ff)
        alldecoys = extract_features(self.decoys,self.ff)
        hit_clusts,_ = cluster_features(allhits,clust_method=clust_method,
                                        max_n=max_n,random_state=random_state)
        self.consensus_hits = compute_feature_centroid_sigma(hit_clusts)
        decoy_clusts,_ = cluster_features(alldecoys,clust_method=clust_method,
                                          max_n=max_n,random_state=random_state)
        self.consensus_decoys = compute_feature_centroid_sigma(decoy_clusts)

    def train(self,dist_thresh=1.5,potency_thresh=0.1,clust_method='hierarchical',max_n=10,random_state=None):
        self.prepare_training_set(dist_thresh=dist_thresh,potency_thresh=potency_thresh)
        self.generate_training_features(self,clust_method=clust_method,max_n=max_n,random_state=random_state)

    def read_ph4(self,ph4_file):
        pass

### Things below this point are likely not useful
    

def ph4_from_MOE(moe):
    '''
    Extract rdkit-compatible pharmacophores from MOE-format input
    Args:
    moe: MOE description of pharmacophore
    Returns:
    essential_ph4: rdkit pharmacophore containing only features marked as "essential" in MOE
    full_ph4: rdkit pharmacophore containing all features in moe
    essential_radii: list of feature radii for essential features, as defined in MOE
    all_radii: list of feature radii for all features, as defined in MOE
    '''
    ph4Info = {}
    header = []
    content = []
    for line in moe.split("\n"):
        if line[0] == "#":
            if len(header) > 0:
                ph4Info[header[0]] = (header,content)
                header = []
                content = []
            header = line.strip().split()
        else:
            content.extend(line.strip().split())
    essential_feats = []
    non_essential_feats = []
    essential_radii = []
    non_essential_radii = []
    type_conversions = {"Acc":"Acceptor","Don":"Donor","Aro":"Aromatic","Hyd":"Hydrophobe",
                       "Ani":"NegIonizable","Cat":"PosIonizable"}
    header, content = ph4Info['#feature']
    num_feats = int(header[1])
    len_feature = int((len(header)/2)-1)
    for i in range(num_feats):
        essential = content[(len_feature*i)+6]
        feat_type = content[(len_feature*i)]
        (x,y,z) = (float(content[(len_feature*i)+2]),float(content[(len_feature*i)+3]),
                   float(content[(len_feature*i)+4]))
        r = float(content[(len_feature*i)+5])
        if essential == '1':
            essential_feats.append(ChemicalFeatures.FreeChemicalFeature(type_conversions[feat_type],
                                                                       Geometry.Point3D(x,y,z)))
            essential_radii.append(r)
        else:
            non_essential_feats.append(ChemicalFeatures.FreeChemicalFeature(type_conversions[feat_type],
                                                                       Geometry.Point3D(x,y,z)))
            non_essential_radii.append(r)
    essential_ph4 = Pharmacophore.Pharmacophore(essential_feats)
    all_feats = essential_feats+non_essential_feats
    full_ph4 = Pharmacophore.Pharmacophore(all_feats)
    all_radii = essential_radii + non_essential_radii
    
    return essential_ph4, full_ph4, essential_radii, all_radii

def mols_from_sdf(supp_file):
    '''
    Helper function to extract molecules from sdf file, with explicit hydrogens
    Args:
    supp_file: .sdf file containing molecules
    Returns:
    mols: list of molecules from supp_file, with explicit hydrogens added
    '''
    with Chem.SDMolSupplier(supp_file,removeHs=False) as supplier:
        mols = [Chem.AddHs(x) for x in supplier if x is not None] 
        # add explicit hydrogens for ease of later conformer generation
    return mols

def molecule_from_smiles_3D(smiles,embed=False):
    '''
    Helper function to generate molecules with optional 3D embedding from SMILES string
    Args:
    smiles: SMILES string to generate molecule from
    embed: whether to generate a 3D embedding of the molecule
    Returns:
    molecule: RDKit molecule object corresponding to smiles, with explicit hydrogens.
        if embed==True, also contains a 3D embedding
    '''
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule) # downstream RDKit functions like having explicit hydrogens, so this is hardcoded
    if embed: # generate 3D embedding if desired
        AllChem.EmbedMolecule(molecule)
        AllChem.MMFFOptimizeMolecule(molecule)
    return molecule

def fuzzify_ph4(ph4, rad_dict = {},default_radius=1.08265):
    '''
    Edits bounds matrix of a pharmacophore object to allow for positional uncertainty
    Args:
    ph4: pharmacophore object to operate on
    rad_dict: dictionary with radii [Angstroms] for each feature type.
        feature types present in ph4 but not rad_dict will be assigned default radius
    default_radius: radius to assign to feature types not explicitly listed in rad_dict
        default value of 1.08265 taken from https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-3-13
    Returns:
    ph4_fuzzy: pharmacophore object with edited bounds matrix
    '''
    
    # initialize radii for any feature types not provided
    features = ph4.getFeatures()
    unique_types = pd.unique(pd.Series([feature.GetFamily() for feature in features]))
    for family in unique_types:
        if family not in rad_dict:
            rad_dict[family] = default_radius
#     print(rad_dict)
    
    # get feature type for each feature and create mapping
    feature_types = {}
    for i in range(len(features)):
        feat_type = features[i].GetFamily()
        feature_types[i] = feat_type
#     print(feature_types)

    # update bounds matrix
    ph4_fuzzy = Pharmacophore.Pharmacophore(features)
    for i in range(len(features)):
        i_type = feature_types[i]
        for j in range(i+1, len(features)):
            j_type = feature_types[j]
            max_uncertainty = rad_dict[i_type]+rad_dict[j_type]
            ph4_fuzzy.setLowerBound(i,j,max(ph4_fuzzy.getLowerBound(i,j)-max_uncertainty,0))
            ph4_fuzzy.setUpperBound(i,j,ph4_fuzzy.getUpperBound(i,j)+max_uncertainty)

    return ph4_fuzzy
    
def ph4_from_molecule(molecule,feature_factory=None,minimal=False,fuzzy=True,rad_dict={},default_radius=1.08265):
    '''
    Generate pharmacophore model from a RDKit molecule
    Args:
    molecule: query molecule to generate pharmacophore for
    feature_factory: rdkit feature factory containing ph4 feature definitions.
        if None, will initialize default feature factory that was installed with rdkit
    minimal: whether to generate a minimal ph4 that can be used for alignment, 
        or the full ph4 with all identified features
    fuzzy: whether to add positional uncertainty to the bounds matrix.
        only set to False if you only need centroids of pharmacophore features
    rad_dict: dictionary of radii [Angstroms] for each ph4 feature type.
        will set radii of any features not explicitly included to a default value
        not used if fuzzy=False
    default_radius: default radius to be passed to fuzzify_ph4. not used if fuzzy=False
    Returns:
    ph4: rdkit pharmacophore model derived from input smiles
    features: list of RDKit features used to create ph4
    '''
    # get default feature factory if one wasn't provided
    if feature_factory == None:
        feature_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    
    # extract pharmacophore features from molecule and create pharmacophore
    features = list(feature_factory.GetFeaturesForMol(molecule))
    # prune feature list to avoid overlaps later
    bad_feats = []
    # things to prune:
    ## remove hydrophobes/lumped hydrophobes that are already counted in aromatic rings
    ## RDKit default h-bond donor definition is a little promiscuous, 
    ##     so remove donors that are also ID'ed as acceptors
    ## remove rings that contain other ph4 features (this one in particular is possibly a bad idea,
    ##     but RDKit struggles with multi-member ph4s that contain other features)
    aromatic_list = []
    acceptor_list = []
    single_atoms = []
    for feat in features:
        if feat.GetFamily().lower() == 'aromatic':
            aromatic_list.append(feat.GetAtomIds())
        elif feat.GetFamily().lower() == 'acceptor' and minimal:
            acceptor_list.append(feat.GetAtomIds())
        if len(feat.GetAtomIds()) == 1 and feat.GetFamily().lower()!='hydrophobe' and minimal:
            single_atoms.append(feat.GetAtomIds()[0])
    single_atoms = np.unique(single_atoms)
    for feat in features:
        if feat.GetFamily().lower() == 'hydrophobe':
            atom = feat.GetAtomIds()[0]
            if any(atom in aromatic for aromatic in aromatic_list):
                bad_feats.append(feat.GetId())
        elif feat.GetFamily().lower() == 'lumpedhydrophobe':
            atoms = feat.GetAtomIds()
            if atoms in aromatic_list:
                bad_feats.append(feat.GetId())
        elif feat.GetFamily().lower() == 'donor' and minimal:
            atom = feat.GetAtomIds()[0]
            if any(atom in acc for acc in acceptor_list):
                bad_feats.append(feat.GetId())
        elif len(feat.GetAtomIds()) > 1 and minimal:
            atoms = feat.GetAtomIds()
            if any(a in single_atoms for a in atoms):
                bad_feats.append(feat.GetId())
    # adjust bad feat indices, since feat.GetId() is 1-indexed
    bad_feats = [x-1 for x in bad_feats]
    # remove bad features
    for i in sorted(bad_feats,reverse=True):
        del features[i]
    ph4 = Pharmacophore.Pharmacophore(features)
    
    if fuzzy:
        ph4 = fuzzify_ph4(ph4,rad_dict=rad_dict,default_radius=default_radius)
    
    return ph4, features

def ph4_from_smiles(smiles, feature_factory=None, minimal=False, fuzzy=True, rad_dict={},default_radius=1.08265):
    '''
    Generate pharmacophore model from a SMILES string
    Args:
    smiles: query SMILES to generate pharmacophore for
    feature_factory: rdkit feature factory containing ph4 feature definitions.
        if None, will initialize default feature factory that was installed with rdkit
    minimal: whether to generate a minimal ph4 that can be used for alignment, 
        or the full ph4 with all identified features
    fuzzy: whether to add positional uncertainty to the bounds matrix.
        only set to False if you only need centroids of pharmacophore features
    rad_dict: dictionary of radii [Angstroms] for each ph4 feature type.
        will set radii of any features not explicitly included to a default value
        not used if fuzzy=False
    default_radius: default radius to be passed to fuzzify_ph4. not used if fuzzy=False
    Returns:
    molecule: rdkit molecule object derived from input smiles
    ph4: rdkit pharmacophore model derived from input smiles
    features: RDKit chemical features used to make ph4
    '''
    # create 3D molecule structure
    molecule = molecule_from_smiles_3D(smiles,embed=True)
    
    # get pharmacophore model and list of features from molecule
    ph4, features = ph4_from_molecule(molecule, feature_factory=feature_factory,minimal=minimal,
                                     fuzzy=fuzzy,rad_dict=rad_dict,default_radius=default_radius)
    return molecule, ph4, features

def get_transform(ref,conformer,atom_match):
    align = []
    for match in atom_match:
        point = Geometry.Point3D(0.0,0.0,0.0)
        for atom in match:
            point+=conformer.GetAtomPosition(atom)
        point/=len(match)
        align.append(point)
    return (rdAlignment.GetAlignmentTransform(ref,align))

def align_mol_to_ph4(molecule,ph4,feature_factory=None,embed_count=100,optimize_energies=True,verbose=False):
    '''
    Align a molecule to a target pharmacophore, if possible
    Args:
    molecule: candidate molcule to be aligned (make sure hydrogens have been added!)
    ph4: target pharmacophore to align to
    feature_factory: feature factory used to find features in molecule. initializes default factory if None
    embed_count: number of embeddings to generate (default: 100)
    optimize_energies: whether to run energy minimization on each embedding prior to aligning to ph4.
    verbose: whether to print progress messages
    Returns:
    min(deviations): minimum SSD between the centroid of each pharmacophore feature and the atom (or centroid of atoms)
        that matches it in the query molecule
    best_embedding: embedding of molecule with lowest SSD to target pharmacophore
    '''
    
    # initialize default feature factory if needed
    if feature_factory == None:
        feature_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    
    # quickly check if alignment is possible based on features in molecule
    match_pass, matches = EmbedLib.MatchPharmacophoreToMol(molecule,feature_factory,ph4)
    if not match_pass:
        if verbose:
            print("Molecule does not have required features to match pharmacophore")
        return "match", None, None
    elif verbose:
        print("Molecule features can match pharmacophore; proceeding to embed")
    
    # use bounds matrix to match molecule to pharmacophore and embed
    bounds_matrix = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    geo_match_fail,bounds_matrix_matched,matched,match_details = EmbedLib.MatchPharmacophore(matches,bounds_matrix,
                                                                                            ph4,useDownsampling=True)
    if geo_match_fail==1:
        if verbose:
            print("Bounds matrix-based matching failed")
        return "bounds", None, None
    atom_match = [f.GetAtomIds() for f in matched]
#     print(atom_match)
    adj_bounds_matrix,embeddings,num_failed = EmbedLib.EmbedPharmacophore(molecule,atom_match,ph4,
                                                                          count=embed_count)
    if len(embeddings) == 0:
        if verbose:
            print("All embeddings failed")
        return "embed_fail", None, None
    elif verbose:
        print(f"{len(embeddings)} embeddings created successfully; proceeding to align")
    
    # optimize energies of successful embeddings
    if optimize_energies:
        for mol in embeddings:
            e1,e2 = EmbedLib.OptimizeMol(mol,adj_bounds_matrix,atomMatches=atom_match)
    
    # compute alignments to pharmacophore
    ref = [feature.GetPos() for feature in ph4.getFeatures()] # get reference locations to align to
    deviations = []
    for mol in embeddings:
        conf = mol.GetConformer()
        dev,transform = get_transform(ref,conf,atom_match)
        rdMolTransforms.TransformConformer(conf,transform)
        deviations.append(dev)
    if verbose:
        print("Alignment complete")
    
    # extract best conformer (minimized distance deviation)
    best_index = np.argmin(deviations)
    best_embedding = embeddings[best_index]
    if verbose:
        print(f"Minimum SSD: {min(deviations)}")
        print("Done")
        
    min_dev = deviations[best_index]
    
    return 0, min_dev, best_embedding

def score_ph4_alignment(conf,ph4,method="tversky",tversky_alpha=0.95,ph4_radius=1.08265,feature_factory=None):
    '''
    Score a molecule alignment to a target pharmacophore based on Gaussian volume overlap
    Args:
    conf: embedding of the query molecule to be scored
    ph4: target pharmacophore to be scored against
    method: which formula to use for score calculation (tversky or tanimoto)
    tversky_alpha: value of alpha for calculating tversky similarity (range 0 to 1)
        lower values bias towards molecules that are subsets of the target pharmacophore;
        higher values bias towards molecules that are supersets of the target pharmacophore.
        not used if method == 'tanimoto'
    ph4_radius: width for Gaussian volumes of pharmacophore features.
        default value taken from https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-3-13
    feature_factory: rdkit feature factory with definitions of pharmacophore features
        should be the same feature factory as was used to generate ph4
        if None, initializes the default rdkit feature factory
    Returns:
    score: volume overlap score between query conformer and target pharmacophore
    '''
    
    # initialize feature factory if required
    if feature_factory == None:
        feature_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    query_feats = list(feature_factory.GetFeaturesForMol(conf))
    # remove hydrophobes that are already identified as part of aromatic rings
    ## (RDKit default definitions have a tendency to double-count these)
    bad_feats = []
    aromatic_list = []
    for feat in query_feats:
        if feat.GetFamily().lower() == 'aromatic':
            aromatic_list.append(feat.GetAtomIds())
    for feat in query_feats:
        if feat.GetFamily().lower() == 'hydrophobe':
            atom = feat.GetAtomIds()[0]
            if any(atom in aromatic for aromatic in aromatic_list):
                bad_feats.append(feat.GetId())
        elif feat.GetFamily().lower() == 'lumpedhydrophobe':
            atoms = feat.GetAtomIds()
            if atoms in aromatic_list:
                bad_feats.append(feat.GetId())
    # adjust bad feat indices, since feat.GetId() is 1-indexed
    bad_feats = [x-1 for x in bad_feats]
    # remove bad features
    for i in sorted(bad_feats,reverse=True):
        del query_feats[i]
    # calculate Gaussian overlaps of ph4 features
    ref_feats = ph4.getFeatures()
    alpha = (np.pi*(3*np.sqrt(2)/(2*np.pi))**(2/3))*ph4_radius**(-2) 
        # calculated as per https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21307
    rq_overlaps = np.zeros(len(ref_feats))
    rr_overlaps = np.zeros(len(ref_feats))
    counter = 0
    for rf in ref_feats:
        # calculate reference-query overlap volumes
        # only consider features of same type as reference feature under consideration
        ref_pos = np.array(rf.GetPos())
        family = rf.GetFamily()
        query_subset = [qf for qf in query_feats if qf.GetFamily()==family]
        query_pos = [np.array(qf.GetPos()) for qf in query_subset]
        distances = np.array([np.linalg.norm(qp-ref_pos) for qp in query_pos])
        overlaps = 2*2.7*(np.pi/(2*alpha))**(3/2)*np.exp(-alpha*alpha*distances/(2*alpha))
            # calculated as per https://pubs.acs.org/doi/10.1021/acs.jcim.4c00516
        rq_overlaps[counter] = np.sum(overlaps)
        # also calculate reference-reference overlap volumes
        ref_subset = [rf2 for rf2 in ref_feats if rf2.GetFamily()==family]
        ref_pos_set = [np.array(rf2.GetPos()) for rf2 in ref_subset]
        ref_distances = np.array([np.linalg.norm(rp-ref_pos) for rp in ref_pos_set])
        ref_overlap = 2*2.7*(np.pi/(2*alpha))**(3/2)*np.exp(-alpha*alpha*ref_distances/(2*alpha))
        rr_overlaps[counter] = np.sum(ref_overlap)
        counter+=1
    qq_overlaps = np.zeros(len(query_feats))
    counter=0
    for qf in query_feats:
        # calculate self-overlap for query molecule
        pos = np.array(qf.GetPos())
        family = qf.GetFamily()
        query_subset = [qf2 for qf2 in query_feats if qf2.GetFamily()==family]
        query_pos = [np.array(qf2.GetPos()) for qf2 in query_subset]
        distances = np.array([np.linalg.norm(qp-pos) for qp in query_pos])
        overlaps = 2*2.7*(np.pi/(2*alpha))**(3/2)*np.exp(-alpha*alpha*distances/(2*alpha))
        qq_overlaps[counter] = np.sum(overlaps)
        counter+=1

    # calculate score
    if method == 'tversky':
        score = np.sum(rq_overlaps)/(tversky_alpha*np.sum(rr_overlaps)+(1-tversky_alpha)*np.sum(qq_overlaps))
    elif method == 'tanimoto':
        score = np.sum(rq_overlaps)/(np.sum(rr_overlaps)+np.sum(qq_overlaps)-np.sum(rq_overlaps))
    return score