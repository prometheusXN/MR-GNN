"""
dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

def load_interaction_data(filename= "cci900_cross" ,featurizer='GraphConv', split='index',frac_train=.72,frac_test=.1):
    """Load Tox21 datasets. Does not do train/test split"""
    # Featurize dataset with Smile strings
    print("About to featurize dataset.")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(
        current_dir, "../datasets/"+ filename +".csv.zip")
    tasks = ['text_mining']
    if featurizer == 'ECFP':
        featurizer_func = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer_func = dc.feat.ConvMolFeaturizer()
    loader1 = dc.data.CSVLoader(tasks = tasks, smiles_field="smiles1", featurizer=featurizer_func)
    loader2 = dc.data.CSVLoader(tasks = tasks, smiles_field="smiles2", featurizer=featurizer_func)
    dataset1 = loader1.featurize(dataset_file, shard_size = 8192)
    dataset2 = loader2.featurize(dataset_file, shard_size = 8192)
    # dataset = [dataset1.X, dataset1.ids, dataset2.X, dataset2.ids, dataset1.y]

    splitters = {'index': dc.splits.IndexSplitter(),
                 'random': dc.splits.RandomSplitter(),
                 'scaffold': dc.splits.ScaffoldSplitter(),
                 'butina': dc.splits.ButinaSplitter()}
    splitter = splitters[split]
    train1, valid1, test1 = splitter.train_valid_test_split(dataset1,frac_train=frac_train, frac_valid=1-frac_train-frac_test, frac_test=frac_test)
    train2, valid2, test2 = splitter.train_valid_test_split(dataset2,frac_train=frac_train, frac_valid=1-frac_train-frac_test, frac_test=frac_test)

    train = [train1, train2]
    valid = [valid1, valid2]
    test = [test1, test2]
    return tasks, (train, valid, test)
