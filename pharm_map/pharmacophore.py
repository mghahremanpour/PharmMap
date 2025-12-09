import numpy as np
import rdkit
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDConfig, Geometry, RDLogger, DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, rdDistGeom, rdMolTransforms, rdShapeAlign, Draw, rdChemicalFeatures, rdMolAlign
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Numerics import rdAlignment
from rdkit.Chem.Draw import IPythonConsole, MolDrawing
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.FeatMaps import FeatMapUtils, FeatMaps
from rdkit.ML.Cluster import Butina
import os
import itertools
import copy
import random
import scipy
from scipy.spatial.transform import Rotation as R
import py3Dmol
import pickle
import sklearn as skl
import umap
import time

DrawingOptions.includeAtomNumbers=True
def timer(func):
    """
    This decorator reports the execution time 
    of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result     = func(*args, **kwargs)
        end_time   = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds",flush=True)
        return result
    return wrapper

def optimal_kmeans(data,min_k=2,max_k=10,random_state=None):
    '''
    Determines optimal k for k-means clustering and performs clustering at optimal value
    Args:
    data: matrix of data to cluster
    min_k, max_k: min and max values of k to sweep over
    random_state: seed for randomization
    Returns:
    opt_clusts: cluster assignments for optimal value of k
    best_k: k value that maximizes silhouette score
    max_silhouette: silhouette score of optimal k
    silhouettes: list of silhouette scores for each k value attempted
    '''
    max_silhouette = -1
    silhouettes = []
    best_k=min_k
    # iterate over all ks and find maximum silhouette score
    for k in range(min_k,max_k):
        clusterer = skl.cluster.KMeans(n_clusters=k,random_state=random_state)
        clust_labels = clusterer.fit_predict(data)
        silhouette_avg = skl.metrics.silhouette_score(data,clust_labels)
        silhouettes.append(silhouette_avg)
        if silhouette_avg>max_silhouette:
            best_k = k
            max_silhouette = silhouette_avg
    # run optimal clustering
    opt_clusterer = skl.cluster.KMeans(n_clusters=best_k,random_state=random_state)
    opt_clusts = opt_clusterer.fit_predict(data)
    return opt_clusts, best_k, max_silhouette, silhouettes

def optimal_hierarchical_clustering(data,min_nclust,max_nclust):
    '''
    Determines optimal n for hierarchical clustering and performs clustering at optimal value
    Args:
    data: matrix of data to cluster
    min_nclust, max_nclust: min and max values of n to sweep over
    Returns:
    opt_clusts: cluster assignments for optimal value of n
    best_n: number of clusters that maximizes silhouette score
    max_silhouette: silhouette score of optimal n
    silhouettes: list of silhouette scores for each n value attempted
    '''
    max_silhouette = -1
    silhouettes = []
    best_n=min_nclust
    # iterate over all n and find maximum silhouette score
    for n in range(min_nclust,max_nclust):
        clusterer = skl.cluster.AgglomerativeClustering(n_clusters=n)
        clust_labels = clusterer.fit_predict(data)
        silhouette_avg = skl.metrics.silhouette_score(data,clust_labels)
        silhouettes.append(silhouette_avg)
        if silhouette_avg>max_silhouette:
            best_n = n
            max_silhouette = silhouette_avg
    # run optimal clustering
    opt_clusterer = skl.cluster.AgglomerativeClustering(n_clusters=best_n)
    opt_clusts = opt_clusterer.fit_predict(data)
    return opt_clusts, best_n, max_silhouette, silhouettes
                
def get_representative_conformers(mol,thresh=1.5,make_mols=False):
    '''
    Generate set of representative conformers from an input molecule
    Args:
    mol: molecule (with conformers already generated) to make conf set from
    thresh: distance threshold [Angstroms] beyond which conformers are not considered neighbors
    make_mols: whether or not to return new rdMol objects
    Returns:
    new_mols: list of rdMol objects with one conformer each if make_mols==True, or None if make_mols==False
    '''
    # align conformers and get RMSDs
    dist_matrix = AllChem.GetConformerRMSMatrix(mol)
    cids = [c.GetId() for c in mol.GetConformers()]
    # cluster conformers and extract centroids; cluster centroids are representative conformers
    clusts = Butina.ClusterData(dist_matrix,len(cids),distThresh=thresh,isDistData=True,reordering=True)
    centroids = [x[0] for x in clusts]
    centroid_cids = [cids[i] for i in centroids]
    if make_mols: # make new single-conformer rdMol objects if desired
        new_mols=[]
        for c in centroid_cids:
            nm = Chem.Mol(mol,confId=c)
            new_mols.append(nm)
    else:
        new_mols=None
    return clusts, centroid_cids, new_mols

def extract_features(mols,feature_factory=None):
    '''
    Extract pharmacophoric features from a set of molecules
    Args:
    mols: list of rdMol objects to get features from
    feature_factory: rdkit feature factory built from definitions of features of interest.
        will use rdkit default if None provided
    Returns:
    feat_df: pd.DataFrame containing feature families and x,y,z positions
    '''
    # get rdkit default feature factory if none was provided
    if feature_factory == None:
        feature_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        feature_factory = ChemicalFeatures.BuildFeatureFactory(feature_path)
    families = []
    xs = []
    ys = []
    zs = []
    end_index=0
    # loop over molecules and extract ph4 features
    for m in mols:
        feats = feature_factory.GetFeaturesForMol(m)
        aromatic_positions = []
        for f in feats:
            families.append(f.GetFamily())
            xs.append(f.GetPos().x)
            ys.append(f.GetPos().y)
            zs.append(f.GetPos().z)
        # some kludge to get around rdkit's bad feature definitions
        # (removing lumped hydrophobes that overlap with aromatics)
            if f.GetFamily().lower()=='aromatic':
                aromatic_positions.append((f.GetPos().x,f.GetPos().y,f.GetPos().z))
        bad_indices = []
        for i in range(len(feats)):
            f = feats[i]
            if f.GetFamily().lower()=='lumpedhydrophobe':
                pos = (f.GetPos().x,f.GetPos().y,f.GetPos().z)
                if pos in aromatic_positions:
                    bad_indices.append(i)
        bad_indices = [x+end_index for x in bad_indices]
        for i in sorted(bad_indices,reverse=True):
            del families[i]
            del xs[i]
            del ys[i]
            del zs[i]
        end_index=len(families)
    # pack data into pd.DataFrame
    feat_df = pd.DataFrame({"Family":families,'x':xs,'y':ys,'z':zs})
    return feat_df

def align_conformers(mols,ref_mol=None):
    '''
    Canonicalize all conformers of input molecules and align them
    Args:
    mols: iterable of rdMol objects
    ref_mol: reference molecule to align to. will use first molecule in mols if none provided
    Returns:
    RMSDs: list of RMSD values for optimal alignments. lower indicates better alignment
    scores: list of scores produced by O3A for optimal alignments. higher indicates better alignment
    '''
    if ref_mol==None:
        ref_mol = mols[0]
    # extract molecule properties for aligner
    ref_params = AllChem.MMFFGetMoleculeProperties(ref_mol)
    RMSDs = []
    scores = []
    # loop over query molecules to align
    for m in mols:
        # canonicalize conformers first to have consistent starting point
        rdMolTransforms.CanonicalizeMol(m)
        mmff_params=AllChem.MMFFGetMoleculeProperties(m)
        mol_scores={}
        mol_RMSDs = {}
        # get Aligner objects for each conformer
        aligners = rdMolAlign.GetO3AForProbeConfs(m,ref_mol,1,mmff_params,ref_params)
        # loop over conformers and run alignments
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

def cluster_features(feats,clust_method='hierarchical',max_n=10,random_state=None,verbose=False):
    '''
    Cluster pharmacophore features
    Args:
    feats: pd.DataFrame of ph4 features to cluster. column names should include 'Family', 'x', 'y', 'z'
    clust_method: clustering method to use ('gaussian', 'hdbscan', 'hierarchical', or 'k-means')
        'gaussian' uses Bayesian inference to fit a Gaussian mixture model. this is theoretically the preferred option.
        other options use the specified clustering algorithm as implemented in scikit-learn. 'hdbscan' should be faster
        than 'hierarchical' and 'k-means' since it only needs to run once rather than testing a range of parameters
    max_n: maximum number of clusters to consider in screen
        for clust_method='gaussian', this is the maximum number of components in the mixture model. note that
        the model can choose to weight fewer components, so treat this as an upper bound
    random_state: seed for randomization, used for k-means clustering and GMM fitting
    verbose: whether to print progress messages
    Returns:
    clustered_feats: pd.DataFrame of features with cluster IDs added
    clust_qc: tuple of (best_n,max_silhouette,silhouettes) returned by optimal clustering functions'''
    clustered_feats = []
    clust_qc = {}
    # only want to cluster feats within each feature type
    feats_by_fam = feats.groupby('Family')
    for fam, group in feats_by_fam:
        if verbose:
            print(f"Clustering {fam}",flush=True)
        pos_matrix = group[['x','y','z']].to_numpy()
        # reduce max_n if there aren't enough features to cluster
        if max_n>np.shape(pos_matrix)[0]-1:
            max_n = np.shape(pos_matrix)[0]-1
        # run optimal clustering
        if clust_method == 'hierarchical':
            opt_clusts, best_n, max_silhouette, silhouettes = optimal_hierarchical_clustering(pos_matrix,
                                                                                              min_nclust=2,
                                                                                              max_nclust=max_n)
        elif clust_method=='k_means':
            opt_clusts,best_n,max_silhouette,silhouettes = optimal_kmeans(pos_matrix, min_k=2, max_k=max_n,
                                                                          random_state=random_state)
        elif clust_method=='hdbscan':
            clusterer = skl.cluster.HDBSCAN(min_cluster_size=max([5,len(group)/max_n]))
            opt_clusts= clusterer.fit_predict(pos_matrix)
            max_silhouette = skl.metrics.silhouette_score(pos_matrix,opt_clusts)
            silhouettes = [max_silhouette]
            best_n=None
        elif clust_method=='gaussian':
            #TODO: make covariance type for this and gaussian_twostage user-specifiable
            clusterer = skl.mixture.BayesianGaussianMixture(n_components=max_n,covariance_type='spherical',
                                                            random_state=random_state,max_iter=2000,
                                                            init_params='k-means++')
            # using spherical covariance, since consensus features will later be modelled as spherical Gaussians
            opt_clusts = clusterer.fit_predict(pos_matrix)
            if len(np.unique(opt_clusts))>1:
                max_silhouette = skl.metrics.silhouette_score(pos_matrix,opt_clusts)
            else:
                max_silhouette=None
            silhouettes=[max_silhouette]
            best_n=len(np.unique(opt_clusts)) # number of components that had features assigned to them, which may be less than max_n
        elif clust_method=='gaussian_twostage':
            clusterer = skl.mixture.BayesianGaussianMixture(n_components=max_n,covariance_type='spherical',
                                                            random_state=random_state,max_iter=1000)
            opt_clusts2 = clusterer.fit_predict(pos_matrix)
            nclusts = len(np.unique(opt_clusts2)) # get the number of clusters the model actually used
            # refit a mixture model with fixed number of components
            clusterer2 = skl.mixture.GaussianMixture(n_components=nclusts,covariance_type='spherical',
                                                     random_state=random_state,max_iter=1000)
            opt_clusts = clusterer2.fit_predict(pos_matrix)
            if len(np.unique(opt_clusts))>1:
                max_silhouette = skl.metrics.silhouette_score(pos_matrix,opt_clusts)
            else:
                max_silhouette=None
            silhouettes=[max_silhouette]
            best_n=len(np.unique(opt_clusts))
            #TODO: allow retention of skl.GaussianMixture object, for predict_proba
        # assign cluster IDs to features
        group['Cluster'] = opt_clusts
        if len(clustered_feats)==0:
            clustered_feats = [group]
        else:
            clustered_feats.append(group)
        # package QC information
        clust_qc[fam] = (best_n,max_silhouette,silhouettes)
    clustered_feats = pd.concat(clustered_feats).reset_index(drop=True)
    return clustered_feats, clust_qc

def compute_feature_centroid_sigma(feats,default_radius=1.08265):
    '''
    Compute centroid and Gaussian sigma for consensus feature clusters
    Args:
    feats: pd.DataFrame of ph4 features, with cluster IDs
    default_radius: radius of a single "color atom"; default value from PubChem3D (see https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-3-13)
    Returns:
    consensus_feats: pd.DataFrame of consensus features, with (x,y,z) position set to cluster centroids
        and Gaussian sigma reported'''
    consensus_feats = []
    # extract feature clusters by family
    feats_by_fam = feats.groupby('Family')
    default_sigma = (np.pi*(3*np.sqrt(2)/(2*np.pi))**(2/3))*default_radius**(-2)
    # calculated as per https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21307
    for fam, group in feats_by_fam:
        clusts = group.groupby('Cluster')
        for clust, g in clusts:
            n_feats = len(g)
            # only do centroid and standard deviation calculation if cluster has more than one feature
            if n_feats > 1:
                pos_matrix = g[['x','y','z']].to_numpy()
                centroid = np.average(pos_matrix,axis=0)
                # calculate standard deviation from covariance matrix
                # treat each consensus feature as a sphere, so use average
                cov = np.cov(pos_matrix.T)
                gauss_sigma = np.average(np.diag(cov))
                sigma = np.sqrt(gauss_sigma**2+default_sigma**2) # add variance of a single "color atom"; value from PubChem3D
            else:
                centroid = g[['x','y','z']].to_numpy()[0]
                sigma = default_sigma # default value used by PubChem3D
                ## see https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-3-13
            #TODO: remove clusters with extremely large variances
            consensus_clust = pd.DataFrame({'Family':[fam],'x':[centroid[0]],'y':[centroid[1]],
                                            'z':[centroid[2]],'sigma':[sigma],'n_feats':[n_feats]})
            consensus_feats.append(consensus_clust)
    consensus_feats = pd.concat(consensus_feats).reset_index(drop=True)
    return consensus_feats

class PharmMapper:
    ff_path = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    ff=ChemicalFeatures.BuildFeatureFactory(ff_path)
    hits=[]
    decoys=[]
    scaffold=None
    all_feats=None
    classifier=None
    scaler=None

    def __init__(self,train_mols,test_mols=None,feature_factory=None,potency_key='IC50'):
        self.train_mols=train_mols
        self.test_mols = test_mols
        if feature_factory: #overwrite default feature factory if a different one was provided
            self.ff = feature_factory
        self.potkey = potency_key
        
    @classmethod
    def from_pickle(cls,filename):
        '''
        Reconstruct PharmMapper from a previously pickled PharmMapper
        Args:
        filename: name of pickle file containing PharmMapper information
        Returns:
        mapper: reconstructed PharmMapper
        '''
        with open(filename,'rb') as file:
            ph4_dict = pickle.load(file)
        mapper = cls(ph4_dict['train_mols'],ph4_dict['test_mols'],
                     potency_key=ph4_dict['potkey'])
        if 'all_feats' in ph4_dict.keys():
            mapper.all_feats=ph4_dict['all_feats']
            mapper.consensus_hits=mapper.all_feats.loc[mapper.all_feats['Class']=='active']
            mapper.consensus_decoys=mapper.all_feats.loc[mapper.all_feats['Class']=='inactive']
        if 'hits' in ph4_dict.keys():
            mapper.hits = ph4_dict['hits']
        if 'decoys' in ph4_dict.keys():
            mapper.decoys = ph4_dict['decoys']
        if 'scaffold' in ph4_dict.keys():
            mapper.scaffold=ph4_dict['scaffold']
        if 'classifier' in ph4_dict.keys():
            mapper.classifier=ph4_dict['classifier']
        if 'scaler' in ph4_dict.keys():
            mapper.scaler = ph4_dict['scaler']
        return mapper
            
    def __unpack_conformers(self,do='train',rep_only=True,dist_thresh=1.5):
        '''
        Unpack rdMol objects with multiple conformers into new rdMol objects with one conformer each
        Args:
        rep_only: whether to return only representative conformers or all conformers
        dist_thresh: maximum neighbor distance for representative conformer clustering
        Returns:
        new_mols: list of rdMol objects with one conformer each
        '''
        new_mols = []
        if do == 'train':
            for m in self.train_mols:
                if len(m.GetConformers())>1:
                    if rep_only:
                        # extract representative conformers if desired
                        _,_,mols_to_use = get_representative_conformers(m,thresh=dist_thresh,
                                                                        make_mols=True)
                        new_mols = new_mols + mols_to_use
                    else:
                        # unpack all conformers into new rdMol objects
                        for c in m.GetConformers():
                            nm=Chem.Mol(m,confId=c.GetId())
                            new_mols.append(nm)
                else:
                    # skip unpacking if molecule already has single conformer
                    new_mols.append(m)
        elif do == 'test':
            for m in self.test_mols:
                if len(m.GetConformers())>1:
                    if rep_only:
                        # extract representative conformers if desired
                        _,_,mols_to_use = get_representative_conformers(m,thresh=dist_thresh,
                                                                        make_mols=True)
                        new_mols = new_mols + mols_to_use
                    else:
                        # unpack all conformers into new rdMol objects
                        for c in m.GetConformers():
                            nm=Chem.Mol(m,confId=c.GetId())
                            new_mols.append(nm)
                else:
                    # skip unpacking if molecule already has single conformer
                    new_mols.append(m)
        return new_mols

    def __split_actives_inactives(self,thresh=0.1):
        '''
        Split training molecules into actives and inactives based on user-set potency threshold
        Args:
        thresh: cutoff value for potency metric (values less than thresh are active, greater are inactive)
        Returns: none'''
        #TODO: if user doesn't specify a threshold, use e.g. histogram analysis to determine one
        for m in self.train_mols:
            if m.GetDoubleProp(self.potkey)<=thresh:
                self.hits.append(m)
            else:
                self.decoys.append(m)

    @timer
    def prepare_mols(self,unpack=True,dist_thresh=1.5,potency_thresh=0.1,rep_only=True,confs='all'):
        '''
        Unpack and align molecules to prepare them for consensus feature identification
        Args:
        dist_thresh: maximum neighbor distance for representative conformer clustering
        potency_thresh: cutoff value for potency metric (values less than thresh are active, greater are inactive)
        rep_only: whether to return only representative conformers or all conformers
        Returns: none
        '''
        # unpack conformers if required
        if unpack:
            self.train_mols = self.__unpack_conformers(do='train',rep_only=rep_only,dist_thresh=dist_thresh)
        # align all training set conformers
        RMSDs,scores = align_conformers(self.train_mols)
        # filter down to only best-aligned conformers, if desired
        if confs == 'best':
            df = pd.DataFrame({'Molecule':self.train_mols,'SMILES':[Chem.MolToSmiles(m) for m in self.train_mols],
                               'RMSD':[r.values()[0] for r in RMSDs]})
            df.sort_values('RMSD',ascending=True,inplace=True)
            best_confs = df.groupby('SMILES',sort=False).head(1)
            self.train_mols = best_confs['Molecule']
        # split training set into actives and inactives
        self.__split_actives_inactives(thresh=potency_thresh)
        # save scaffold molecule for later alignment of test set
        self.scaffold = self.train_mols[0]

    @timer
    def generate_training_features(self,clust_method='gaussian',max_n_hits=10,
                                   max_n_decoys=30,dr=1.08265,random_state=None,
                                   verbose=False):
        '''
        Extract consensus ph4 features from identified hits and decoys
        Args:
        clust_method: clustering method to use ['gaussian' / 'gaussian_twostage' / 'hierarchical' / 'k_means' / 'hdbscan']
        max_n_hits: maximum number of clusters to attempt when clustering features from hits
        max_n_decoys: maximum number of clusters to attempt when clustering features from decoys
        random_state: seed for randomization
        verbose: whether to print progress messages
        Returns: none
        '''
        if verbose:
            print("Extracting active ph4 features",flush=True)
        allhits = extract_features(self.hits,self.ff)
        if verbose:
            print("Extracting inactive ph4 features",flush=True)
        alldecoys = extract_features(self.decoys,self.ff)
        if verbose:
            print("Clustering active features",flush=True)
        hit_clusts,_ = cluster_features(allhits,clust_method=clust_method,
                                        max_n=max_n_hits,random_state=random_state,
                                        verbose=verbose)
        self.consensus_hits = compute_feature_centroid_sigma(hit_clusts,default_radius=dr)
        self.consensus_hits['Class']=['active']*len(self.consensus_hits)
        if verbose:
            print("Clustering inactive features",flush=True)
        decoy_clusts,_ = cluster_features(alldecoys,clust_method=clust_method,
                                          max_n=max_n_decoys,random_state=random_state,
                                          verbose=verbose)
        self.consensus_decoys = compute_feature_centroid_sigma(decoy_clusts,default_radius=dr)
        self.consensus_decoys['Class']=['inactive']*len(self.consensus_decoys)

    def __find_structural_features(self,sim_cutoff):
        '''
        Identify features of high similarity between active and inactive feature sets
        Args:
        sim_cutoff: similarity cutoff above which two features will be considered matched
        Returns: none
        '''
        # initialize distance and similarity matrices
        dists = np.full((len(self.consensus_hits),len(self.consensus_decoys)),np.inf)
        similarities = np.zeros((len(self.consensus_hits),len(self.consensus_decoys)))
        # loop over all feature pairs
        for i in range(len(self.consensus_hits)):
            for j in range(len(self.consensus_decoys)):
                # only consider pairs of same feature type
                if self.consensus_decoys['Family'].iloc[j]==self.consensus_hits['Family'].iloc[i]:
                    # extract feature positions and sigmas
                    pos1 = self.consensus_hits[['x','y','z']].iloc[i].to_numpy()
                    pos2 = self.consensus_decoys[['x','y','z']].iloc[j].to_numpy()
                    sig1 = self.consensus_hits['sigma'].iloc[i]
                    sig2 = self.consensus_decoys['sigma'].iloc[j]
                    # calculate Euclidean distance between centroids
                    dist = np.linalg.norm(pos2-pos1)
                    # calculate Tanimoto similarity of Gaussian volumes
                    overlap = 2*2.7*(np.pi/(sig1+sig2))**(3/2)*np.exp(-sig1*sig2*dist**2/(sig1+sig2))
                    self_overlap1 = 2*2.7*(np.pi/(sig1+sig1))**(3/2)
                    self_overlap2 = 2*2.7*(np.pi/(sig2+sig2))**(3/2)
                    sim = overlap/(self_overlap1+self_overlap2-overlap)
                    # record values
                    similarities[i,j]=sim
                    dists[i,j]=dist
        # find maximum similarity scores for each active feature
        max_sim_indices_1 = np.argmax(similarities,axis=1)
        # mark feature as matched if maximum score is above cutoff
        matched1 = [True if similarities[i,max_sim_indices_1[i]]>sim_cutoff else False 
                    for i in range(len(self.consensus_hits))]
        # record ID of matching feature if applicable
        match_indices_1 = [int(max_sim_indices_1[i]) if matched1[i]==True else None 
                           for i in range(len(self.consensus_hits))]
        # save info to hits DataFrame
        self.consensus_hits['Matched']=matched1
        self.consensus_hits['Match Partner']=match_indices_1
        self.consensus_hits['Maximum Similarity']=np.max(similarities,axis=1).tolist()
        # repeat the above process for decoys
        max_sim_indices_2 = np.argmax(similarities,axis=0)
        matched2 = [True if similarities[max_sim_indices_2[j],j]>sim_cutoff else False 
                    for j in range(len(self.consensus_decoys))]
        match_indices_2 = [int(max_sim_indices_2[j]) if matched2[j]==True else None 
                           for j in range(len(self.consensus_decoys))]
        self.consensus_decoys['Matched']=matched2
        self.consensus_decoys['Match Partner']=match_indices_2
        self.consensus_decoys['Maximum Similarity']=np.max(similarities,axis=0).tolist()

    def __assign_scoring_weights(self):
        '''Assign scoring weights to consensus features
            (defaults to number of component features for each consensus feature)
        Args: none
        Returns: none
        '''
        for f in [self.consensus_hits,self.consensus_decoys]:
            # if feature was matched between actives and inactives, set weight to 1 (minimum)
            # else, set weight to number of contributing features
            # more common features within a dataset should count more
            weights=[1 if f['Matched'].iloc[i]==True else f['n_feats'].iloc[i]
                     for i in range(len(f))]
            f['Weight'] = weights
    @timer      
    def calculate_score_matrix(self,mols,default_radius=1.08265,ref_key='all'):
        '''
        Calculate matrix of volumetric overlaps between a set of molecules and reference pharmacophore
        Args:
        mols: query molecules to calculate scores for
        default_radius: radius assigned to "color atoms" - i.e. if the centroid of a pharmacophore feature was
            represented by an atom, it would have this van der Waals radius
        ref_key: which reference features to use to calculate scores ('all', 'active', or 'inactive')
        Returns:
        score_mat: np.array of shape (len(mols),len(features)) containing volumetric overlap scores of each
            query molecule with each reference feature. scores are calculated by Tversky similarity with alpha=1
            (score = v_overlap/v_feature)
        '''
        # get the desired reference features from the consensus map(s)
        if ref_key=='all':
            ref_feats=self.all_feats
        elif ref_key=='active':
            ref_feats=self.consensus_hits
        elif ref_key=='inactive':
            ref_feats=self.consensus_decoys
        # initialize score matrix
        score_mat = np.zeros((len(mols),len(ref_feats)))
        # calculate standard deviation of color atoms
        default_sigma = (np.pi*(3*np.sqrt(2)/(2*np.pi))**(2/3))*default_radius**(-2)
        # helper sub-function to calculate volume overlaps between features of the same type
        def calc_overlaps(row):
            if type(row['Query Positions'])!=bool:
                dists = scipy.spatial.distance.cdist(row[['x','y','z']].to_numpy().reshape(-1,3).astype(float),
                                                     row['Query Positions'])
                overlap = np.sum(2*2.7*(np.pi/(row['sigma']+default_sigma))**(3/2)*
                                np.exp((-default_sigma*row['sigma']*dists**2)/
                                (default_sigma+row['sigma'])))
            else:
                overlap = 0
            return overlap
        # loop over query molecules and calculate overlap scores
        for i in range(len(mols)):
            # extract query features and group by feature type
            feats = extract_features([mols[i]])
            feats_by_group = feats.groupby('Family')
            # get positions of query features that match type of each reference feature
            ref_feats['Query Positions'] = [feats_by_group.get_group(f)[['x','y','z']].to_numpy()
                                            if f in feats_by_group.groups else False
                                            for f in ref_feats['Family']]
            # calculate Gaussian volume overlaps
            ref_feats['Overlaps'] = ref_feats.apply(calc_overlaps,axis=1)
            # calculate Tversky scores of overlaps, using reference feature volume as denominator
            score_mat[i,:] = ref_feats['Overlaps']/(2*2.7*(np.pi/(ref_feats['sigma']*2))**(3/2))
        # clean up reference feature dataframe
        ref_feats.drop('Query Positions',axis=1,inplace=True)
        ref_feats.drop('Overlaps',axis=1,inplace=True)
        return score_mat
    
    @timer
    def make_consensus_ph4(self,unpack=True,dist_thresh=1.5,potency_thresh=0.1,rep_only=True,
              method='hierarchical',max_n_hits=50,max_n_decoys=50,
              sim_cutoff=0.75,dr=1.08265,random_state=None,confs='all',
              verbose=False):
        '''
        Perform consensus feature identification pipeline
        Args:
        dist_thresh: maximum neighbor distance for representative conformer clustering
        potency_thresh: cutoff value for potency metric (values less than thresh are active, greater are inactive)
        rep_only: whether to return only representative conformers or all conformers
        method: clustering method to use ('hierarchical' or 'k_means')
        max_n_hits: maximum number of clusters to attempt when clustering features from hits
        max_n_decoys: maximum number of clusters to attempt when clustering features from decoys
        sim_cutoff: similarity cutoff above which two features will be considered matched
        random_state: seed for randomization
        verbose: whether to print progress messages
        Returns: none
        '''
        if verbose:
            print("Preparing conformers",flush=True)
        self.prepare_mols(unpack=unpack,dist_thresh=dist_thresh,potency_thresh=potency_thresh,
                                  rep_only=rep_only,confs=confs)
        if verbose:
            print(f"{len(self.hits)} active conformers in training set")
            print(f"{len(self.decoys)} inactive conformers in training set")
            print("Generating consensus training features",flush=True)
        self.generate_training_features(clust_method=method,max_n_hits=max_n_hits,
                                        max_n_decoys=max_n_decoys,random_state=random_state,dr=dr,
                                        verbose=verbose)
        if verbose:
            print("Finding overlapping active/inactive features",flush=True)
        self.__find_structural_features(sim_cutoff=sim_cutoff)
        self.__assign_scoring_weights()
        self.all_feats = pd.concat([self.consensus_hits,self.consensus_decoys])

    @timer
    def make_classifier(self,mlp=False,verbose=False):
        '''
        Find the best-performing classifier from a variety of algorithms, and return that fitted classifier
        Args:none
        Returns:
        self.classifier: fitted classifier
        roc: mean ROC-AUC of best-performing classifier, from cross-validation
        results: list of dicts containing cross-validation results, potentially useful for downstream plotting
        '''
        if verbose:
            print("Calculating training set overlap scores",flush=True)
        self.training_scores = self.calculate_score_matrix(self.hits+self.decoys)
        #TODO
        '''
        Possible future addition: use scikit-learn predict_proba on fitted GMM to make scores
        Problem: number of features would be compound-dependent, so how to transform to constant number of features?
        for each consensus feature, take the highest probability among sample features and use that? sum of probabilities?
        This would probably require two-stage GMM fitting to avoid extraneous components with zero weight hanging around
        Also needs to retain the actual GMM object, not just the DataFrame of information, so edit make_consensus_ph4
        Shared active/inactive feature distinctions would potentially be difficult, since they would live in separate skl objects
        Make new GMM from means + variances of identified features after shared detection? using weights_init,means_init,precisions_init
            would still have to call fit() though, so not sure this works
        '''
        self.training_labels = [1]*len(self.hits)+[0]*len(self.decoys)
        # scale scores, since Tversky scores can be >1
        if verbose:
            print("Scaling scores",flush=True)
        self.scaler = skl.preprocessing.StandardScaler()
        scaled_scores = self.scaler.fit_transform(self.training_scores)
        # make Trainer, run cross-validation on range of classifiers and get best one
        if verbose:
            print("Training and cross-validating classifiers",flush=True)
        self.trainer = Trainer(scaled_scores,self.training_labels,do_mlp=mlp)
        self.classifier,auc = self.trainer.find_best_classifier()
        return self.classifier,auc
    
    @timer
    def predict(self,unpack=True,rep_only=True,dist_thresh=1.5,verbose=False,confs='all'):
        '''
        Use a fitted classifier to predict active/inactive probabilities for test set molecules
        Args: none
        Returns:
        self.test_probs: array of shape (len(test_mols),2) containing class probabilities for each test compound
        '''
        # align test conformers to the scaffold used to make consensus features
        if verbose:
            print("Aligning conformers",flush=True)
        self.test_mols = self.__unpack_conformers(do='test',rep_only=rep_only,dist_thresh=dist_thresh)
        RMSDs,scores = align_conformers(self.test_mols,ref_mol=self.scaffold)
        # filter out only best-aligned conformer for each test set molecule, if desired
        if confs == 'best':
            if verbose:
                print("Selecting best-aligned conformers",flush=True)
            df = pd.DataFrame({'Molecule':self.test_mols,'SMILES':[Chem.MolToSmiles(m) for m in self.test_mols],
                               'RMSD':[r.values()[0] for r in RMSDs]})
            df.sort_values('RMSD',ascending=True,inplace=True)
            best_confs = df.groupby('SMILES',sort=False).head(1)
            self.test_mols = best_confs['Molecule']
        # calculate volumetric overlap scores
        if verbose:
            print("Scoring test set",flush=True)
        self.test_overlaps = self.calculate_score_matrix(self.test_mols)
        # scale scores, since Tversky scores can be >1
        if verbose:
            print("Scaling test set scores",flush=True)
        scaled_scores = self.scaler.fit_transform(self.test_overlaps)
        # predict class probabilities using pre-trained classifier
        if verbose:
            print("Predicting test set classes",flush=True)
        self.test_probs = self.classifier.predict_proba(scaled_scores)
        return self.test_probs

    def __make_fps(self):
        '''
        Make molecular fingerprints of all compounds
        '''
        fpgen = AllChem.GetMorganGenerator()
        self.train_fps = [fpgen.GetFingerprint(x) for x in self.train_mols]
        self.test_fps = [fpgen.GetFingerprint(x) for x in self.test_mols]

    
    def calculate_train_test_similarity(self,plot=False,plotfile=None):
        '''
        Calculate maximum Tanimoto similarity between test molecules and training set
        Args:
        plot: whether to plot maximum similarities and save the plot
        plotfile: filename to save plot to; only used if plot=True
        Returns:
        max_scores: list of maximum Tanimoto similarities between each test compound and the training set
        '''
        if not hasattr(self,'train_fps'):
            self.__make_fps()
        self.train_test_sim = np.zeros(len(self.test_fps))
        for i in range(len(self.test_fps)):
            tanimotos = [DataStructs.TanimotoSimilarity(self.test_fps[i], f) for f in self.train_fps]
            max_sim = np.max(tanimotos)
            self.train_test_sim[i] = max_sim
        if plot:
            fig,ax = plt.subplots(1,1)
            ax.hist(self.train_test_sim)
            ax.set_xlabel('Maximum Tanimoto similarity')
            ax.set_ylabel('Counts')
            plt.savefig(plotfile,dpi=300)
        return self.train_test_sim
         
    def save(self,filename):
        '''
        Save mapper information to pickle for later reuse
        Args:
        filename: str name of file to save to
        Returns: none
        '''
        ph4_dict = {}
        if self.scaffold is not None:
            ph4_dict['scaffold']=self.scaffold
        if self.all_feats is not None:
            ph4_dict['all_feats']=self.all_feats
        if self.classifier is not None:
            ph4_dict['classifier']=self.classifier
        if self.scaler is not None:
            ph4_dict['scaler']=self.scaler
        if len(self.hits)>0:
            ph4_dict['hits']=[Chem.PropertyMol(m) for m in self.hits]
        if len(self.decoys)>0:
            ph4_dict['decoys']=[Chem.PropertyMol(m) for m in self.decoys]
        ph4_dict['potkey']=self.potkey
        ph4_dict['train_mols']=[Chem.PropertyMol(m) for m in self.train_mols]
        ph4_dict['test_mols']=[Chem.PropertyMol(m) for m in self.test_mols]
        with open(filename,'wb') as file:
            pickle.dump(ph4_dict,file)
        
    def save_training_feats(self,filepath):
        # save consensus features to pickle for later use
        with open(filepath,'wb') as file:
            pickle.dump(self.all_feats,file)

    def read_ph4(self,ph4_file):
        # load consensus features from pickle and split into actives and inactives
        with open(ph4_file,'rb') as file:
            self.all_feats = pickle.load(file)
        self.consensus_hits=self.all_feats.loc[self.all_feats['Class']=='active']
        self.consensus_decoys=self.all_feats.loc[self.all_feats['Class']=='inactive']
        
class Trainer:
    '''
    Class to handle classifier model selection and training
    '''
    svm = skl.svm.SVC(probability=True)
    svm_params = {'C':np.logspace(-2,3,6,base=2)}
    sgd = skl.linear_model.SGDClassifier(loss='modified_huber',max_iter=10000)
    sgd_params = {'alpha':np.logspace(-5,0,6,base=10)}
    k_neighbors = skl.neighbors.KNeighborsClassifier()
    kn_params = {'leaf_size':np.linspace(10,100,10).astype(int)}
    hgbc = skl.ensemble.HistGradientBoostingClassifier()
    hgbc_params = {'max_leaf_nodes':[11,21,31,41],'min_samples_leaf':[5,10,15,20,25,30]}
    classifiers = [svm,sgd,k_neighbors,hgbc]
    all_params = [svm_params,sgd_params,kn_params,hgbc_params]

    def __init__(self,train_data,train_labels,do_mlp=False):
        self.X = train_data
        self.y = train_labels
        if do_mlp:
            self.mlp = skl.neural_network.MLPClassifier()
            self.mlp_params={'hidden_layer_sizes':[(10,),(25,),(50,),(100,),(250,),(500,)],'alpha':np.logspace(-6,-1,6,base=10)}
            self.classifiers.append(self.mlp)
            self.all_params.append(self.mlp_params)
    
    def parameter_sweep(self,classifier,param_grid,verbose=False):
        '''
        Perform cross-validated parameter sweep on a classifier to optimize hyperparameters
        '''
        scoring = {'AUC':'roc_auc','Precision':'precision','Recall':'recall','Balanced':'balanced_accuracy'}
        if verbose:
            gv = 3
        else:
            gv=0
        g = skl.model_selection.GridSearchCV(classifier,param_grid,scoring=scoring,refit='Balanced',return_train_score=True,verbose=gv)
        g.fit(self.X,self.y)
        return g.best_estimator_,g.best_score_,g.best_params_,g
    
    def train_all_classifiers(self,save_params=True,verbose=False):
        self.trained_models=[]
        self.best_scores = []
        if save_params:
            self.best_params=[]
            self.param_testers=[]
        for i in range(len(self.classifiers)):
            model,score,params,g = self.parameter_sweep(self.classifiers[i],self.all_params[i],verbose=verbose)
            self.trained_models.append(g)
            self.best_scores.append(score)
            if save_params:
                self.best_params.append(params)
                self.param_testers.append(g)

    def find_best_classifier(self,verbose=False):
        '''
        Try a range of classifier models and select the one with best ROC-AUC, using default parameters
        Args: none
        Returns:
        best_classifier: fitted classifier model of type that had best cross-validation performance
        self.best_scores[best_i]: mean cross-validation score of best trained model
        '''
        # self.classifier_opt_results = []
        # self.roc_aucs = []
        # for c in self.classifiers:
        #     results = skl.model_selection.cross_validate(c,self.X,self.y,scoring='roc_auc',
        #                                                  return_estimator=True,return_indices=True)
        #     r_array = results['test_score']
        #     r = np.mean(r_array)
        #     self.roc_aucs.append(r)
        #     self.classifier_opt_results.append(results)
        # best_i = np.argmax(self.roc_aucs)
        # best_classifier = self.classifiers[best_i]
        # self.model = best_classifier
        self.train_all_classifiers(save_params=True,verbose=verbose)
        best_i = np.argmax(self.best_scores)
        self.best_classifier=self.trained_models[best_i]
        return self.best_classifier,self.best_scores[best_i]