'''
CHAMPS Dataset preprocessing Class Alternative Implementation.
(atom_index_1, type) tuple version.

'''

import numpy as np
import pandas as pd
import os
import random

import common as cm


class CHAMPSData:
    def __init__(self, dataset_folder, trainfn, structuresfn, testfn=None, set_features=False, N=None,  p=None, far_dist=None,
                 self_dist=None, save_fn=None, tmpsavepath='./tmp_prep_data'):
        '''
        initialize parameters from CHAMPSData class object

        Inputs:
        dataset_folder: folder path path containing CHAMPS train and test raw data
        trainfn: file name for train csv file from CHAMPS dataset
        structuresfn: file name for structures csv file from CHAMPS dataset
        set_features: if set to True, will prepare data during object initialization
        testfn: if set to True, train and test data will be prepared together.
        N: max number of atoms (nodes) a molecule(graph) can have in dataset.
        p: distance power value, default 2
        far_dist:  distance if two nodes are not connected or too far
        self_dist:  distance for node and itself, typically a very high value
        testfn: if set the train and test data prepped together then separately saved.
        savefn: file name to save the combined preprocessed dataset (X, AdjM, Y, YID)
        tmpsavepath: a folder for saving temporary files during data preparation.

        Returns:
        A CHAMPSData object, and prepares data for the machine learning model.
        '''

        self.datasetpath=os.path.abspath(dataset_folder)
        self.trainpath=os.path.join(self.datasetpath, trainfn)
        self.structpath=os.path.join(self.datasetpath, structuresfn)
        self.dfXtrain = pd.read_csv(self.trainpath)
        self.dfXstructs = pd.read_csv(self.structpath)
        self.N = N if N is not None else (self.dfXtrain['atom_index_1'].nunique())
        self.p = p if p is not None else 2
        self.far_dist = far_dist if far_dist is not None else 1e-5
        self.self_dist = self_dist if self_dist is not None else 1e3
        self.tmpsavepath=tmpsavepath

        if testfn is not None:
            self.testpath = os.path.join(self.datasetpath, testfn)
            self.dfXtest = pd.read_csv(self.testpath)
            self.dfXtest['scalar_coupling_constant']=0

            self.dfXtrain_test = pd.concat([self.dfXtrain, self.dfXtest], keys=['train', 'test'])
        else:
            self.dfXtest=None


        self.comb_fn = save_fn

        if set_features:
            self.prep_champs_data()
        else:
            self.F, self.cls_num, self.dict_adjM = None, None, None
            self.dict_adjMfn=None
            if testfn is None:
                self.X_dict, self.Y_dict, self.YID_dict, self.comb_dict= None, None, None, None
                self.X_dictfn, self.Y_dictfn, self.YID_dictfn, self.comb_dictfn = None, None, None, None
            else:
                self.X_dict_trn, self.Y_dict_trn, self.YID_dict_trn, self.comb_dict_trn = None, None, None, None
                self.X_dict_tst, self.Y_dict_tst, self.YID_dict_tst, self.comb_dict_tst = None, None, None, None

                self.X_dict_trnfn, self.Y_dict_trnfn, self.YID_dict_trnfn, self.comb_dict_trnfn = None, None, None, None
                self.X_dict_tstfn, self.Y_dict_tstfn, self.YID_dict_tstfn, self.comb_dict_tstfn = None, None, None, None




    def info(self):
        '''
        Prints out basic info about object attributes
        '''
        print(self.dfXtrain.info(memory_usage="deep"))
        print(self.dfXstructs.info(memory_usage="deep"))
        if self.dfXtest is None:
            print(pd.DataFrame(self.X_dict).info(memory_usage="deep")) if self.X_dict is not None \
                else print("X_dict is not created")

            print(pd.DataFrame(self.Y_dict).info(memory_usage="deep")) if self.Y_dict is not None \
                else print("Y_dict is not created")

            print(pd.DataFrame(self.YID_dict).info(memory_usage="deep")) if self.YID_dict is not None \
                else print("YID_dict is not created")

            print(pd.DataFrame(self.comb_dict).info(memory_usage="deep")) if self.comb_dict is not None \
                else print("comb_dict is not created")
        else:
            print(self.dfXtest.info(memory_usage="deep"))
            print(self.dfXtrain_test.info(memory_usage="deep"))

            print(pd.DataFrame(self.X_dict_trn).info(memory_usage="deep")) if self.X_dict_trn is not None \
                else print("X_dict is not created")

            print(pd.DataFrame(self.Y_dict_trn).info(memory_usage="deep")) if self.Y_dict_trn is not None \
                else print("Y_dict is not created")

            print(pd.DataFrame(self.YID_dict_trn).info(memory_usage="deep")) if self.YID_dict_trn is not None \
                else print("YID_dict is not created")

            print(pd.DataFrame(self.comb_dict_trn).info(memory_usage="deep")) if self.comb_dict_trn is not None \
                else print("comb_dict is not created")

            print(pd.DataFrame(self.X_dict_trn).info(memory_usage="deep")) if self.X_dict_trn is not None \
                else print("X_dict is not created")

            print(pd.DataFrame(self.Y_dict_tst).info(memory_usage="deep")) if self.Y_dict_tst is not None \
                else print("Y_dict is not created")

            print(pd.DataFrame(self.YID_dict_tst).info(memory_usage="deep")) if self.YID_dict_tst is not None \
                else print("YID_dict is not created")

            print(pd.DataFrame(self.comb_dict_tst).info(memory_usage="deep")) if self.comb_dict_tst is not None \
                else print("comb_dict is not created")

        print(pd.DataFrame(self.dict_adjM).info(memory_usage="deep")) if self.dict_adjM is not None \
            else print("dict_adjM is not created")

        print("N is", self.N)
        print("p is set to", self.p)
        print("far_dist is set to", self.far_dist)
        print("self_dist is set to", self.self_dist)

        print("F is", self.F) if self.F is not None else print("F is not defined")
        print("cls_num is", self.cls_num) if self.cls_num is not None else print("cls_num is not defined")
        print("Prepped data save file name is:", self.comb_fn) if self.comb_fn is not None else \
            print("A file name to save prepared data is not set")

    def make_X(self, dfTrain1):
        '''
        Creates X feauture matrices from dataset.

        Inputs:
        dfTrain1: Train dataframe with atomic_index_ and type are made tuple.

        Returns:
        Saves prepared data in a temporary folder.

        '''

        atom_ind_type='ai1_type'
        atom_ind_node= 'atom_index_0'

        dfTrain2 = pd.get_dummies(dfTrain1[['molecule_name', atom_ind_node, atom_ind_type]], columns=[atom_ind_type])

        print('CREATING THE FEATURE MATRIX')
        if self.dfXtest is None:
            dfTrain2 = dfTrain2.groupby(['molecule_name', atom_ind_node]).sum()
            self.X_dict = self.fn_matrix(dfTrain2, self.F, self.N)
            print('FEATURE MATRIX DONE', 'dict length is:', len(self.X_dict.keys()),
                  'dict item shape is:', list(self.X_dict.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_X_dict.pkl'), self.X_dict)

            self.X_dictfn='tmp_X_dict.pkl'
            self.X_dict = {}
        else:
            dfTrain3 = dfTrain2.xs('train').groupby(['molecule_name', atom_ind_node]).sum()
            self.X_dict_trn = self.fn_matrix(dfTrain3, self.F, self.N)
            print('FEATURE MATRIX DONE FOR TRAIN, DOING TEST NOW', 'dict length is:', len(self.X_dict_trn.keys()),
                  'dict item shape is:', list(self.X_dict_trn.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_X_dict_trn.pkl'), self.X_dict_trn)

            self.X_dict_trnfn='tmp_X_dict_trn.pkl'
            self.X_dict_trn = {}

            dfTrain3 = dfTrain2.xs('test').groupby(['molecule_name', atom_ind_node]).sum()
            self.X_dict_tst = self.fn_matrix(dfTrain3, self.F, self.N)

            print('FEATURE MATRIX DONE FOR TEST', 'dict length is:', len(self.X_dict_tst.keys()),
                  'dict item shape is:', list(self.X_dict_tst.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_X_dict_tst.pkl'), self.X_dict_tst)

            self.X_dict_tstfn='tmp_X_dict_tst.pkl'
            self.X_dict_tst = {}

    def make_Y(self, dfTrain1):
        '''
        Creates Y label vectors from dataset.

        Inputs:
        dfTrain1: Train dataframe with atomic_index_ and type are made tuple.

        Returns:
        Saves prepared data in a temporary folder.

        '''

        atom_ind_type='ai1_type'
        atom_ind_node= 'atom_index_0'
        print('CREATING SCALAR COUPLING VALUE Y VALUE VECTOR')

        dfTrain2 = pd.get_dummies(
            dfTrain1[['id', 'molecule_name', atom_ind_node, 'scalar_coupling_constant', atom_ind_type]],
            columns=[atom_ind_type])
        dfTrain2.iloc[:, 4:] = dfTrain2.iloc[:, 4:].multiply(dfTrain2.iloc[:, 3], axis='index')

        if self.dfXtest is None:
            dfTrain3 = dfTrain2.iloc[:, np.r_[1, 2, 4:dfTrain2.shape[1]]].groupby(['molecule_name', atom_ind_node]).sum()
            self.Y_dict = self.fn_matrix(dfTrain3, self.F, self.N, isflatten=True)

            print('SCALAR COUPLING CONSTANT Y VECTOR DONE', 'dict length is:', len(self.Y_dict.keys()),
                  'dict item shape is (cls_num):', list(self.Y_dict.values())[0].shape)
            self.cls_num=list(self.Y_dict.values())[0].shape

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_Y_dict.pkl'), self.Y_dict)

            self.Y_dictfn='tmp_Y_dict.pkl'
            self.Y_dict = {}

        else:
            dfTrain3 = dfTrain2.xs('train').iloc[:, np.r_[1, 2, 4:dfTrain2.shape[1]]].groupby(['molecule_name', atom_ind_node]).sum()
            self.Y_dict_trn = self.fn_matrix(dfTrain3, self.F, self.N,
                                         isflatten=True)

            print('SCALAR COUPLING CONSTANT Y VECTOR DONE FOR TRAIN', 'dict length is:', len(self.Y_dict_trn.keys()),
                  'dict item shape is (cls_num):', list(self.Y_dict_trn.values())[0].shape)
            self.cls_num=list(self.Y_dict_trn.values())[0].shape

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_Y_dict_trn.pkl'), self.Y_dict_trn)

            self.Y_dict_trnfn='tmp_Y_dict_trn.pkl'
            self.Y_dict_trn = {}


            print('DUMMMY LABELS FOR TEST DATA ')
            dfTrain3 = dfTrain2.xs('test').iloc[:, np.r_[1, 2, 4:dfTrain2.shape[1]]].groupby(['molecule_name', atom_ind_node]).sum()
            self.Y_dict_tst = self.fn_matrix(dfTrain3, self.F, self.N, isflatten=True)

            print('DUMMY SCALAR COUPLING CONSTANT Y VECTOR DONE FOR TEST', 'dict length is:', len(self.Y_dict_tst.keys()),
                  'dict item shape is (cls_num):', list(self.Y_dict_tst.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_Y_dict_tst.pkl'), self.Y_dict_tst)

            self.Y_dict_tstfn='tmp_Y_dict_tst.pkl'
            self.Y_dict_tst = {}

    def make_YID(self, dfTrain1):
        '''
        Creates Y label id vector from dataset.

        Inputs:
        dfTrain1: Train dataframe with atomic_index_ and type are made tuple.

        Returns:
        Saves prepared data in a temporary folder.

        '''

        atom_ind_type='ai1_type'
        atom_ind_node= 'atom_index_0'
        print('CREATING ID  VECTOR')

        dfTrain2 = pd.get_dummies(
            dfTrain1[['id', 'molecule_name', atom_ind_node, atom_ind_type]],
            columns=[atom_ind_type])

        dfTrain2.iloc[:, 3:] = dfTrain2.iloc[:, 3:].multiply(dfTrain2.iloc[:, 0], axis='index')

        if self.dfXtest is None:
            dfTrain3 = dfTrain2.iloc[:, np.r_[1, 2, 3:dfTrain2.shape[1]]].groupby(
                ['molecule_name', atom_ind_node]).sum()
            self.YID_dict = self.fn_matrix(dfTrain3, self.F, self.N, isflatten=True)

            print('ID VECTOR DONE', 'dict length is:', len(self.YID_dict.keys()),
                  'dict item shape is:', list(self.YID_dict.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath, 'tmp_YID_dict.pkl'), self.YID_dict)

            self.YID_dictfn = 'tmp_YID_dict.pkl'
            self.YID_dict= {}

        else:
            print('CREATING ID  VECTOR FOR TRAIN')
            dfTrain3 = dfTrain2.xs('train').iloc[:, np.r_[1, 2, 3:dfTrain2.shape[1]]].groupby(
                ['molecule_name', atom_ind_node]).sum()
            self.YID_dict_trn = self.fn_matrix(dfTrain3, self.F, self.N,
                                           isflatten=True)  # ids as flat vector, keys are molecule names
            print('ID VECTOR DONE', 'dict length is:', len(self.YID_dict_trn.keys()),
                  'dict item shape is:', list(self.YID_dict_trn.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath,'tmp_YID_dict_trn.pkl'), self.YID_dict_trn)

            self.YID_dict_trnfn='tmp_YID_dict_trn.pkl'
            self.YID_dict_trn = {}

            print('CREATING ID  VECTOR FOR TEST')
            dfTrain3 = dfTrain2.xs('test').iloc[:, np.r_[1, 2, 3:dfTrain2.shape[1]]].groupby(
                ['molecule_name', atom_ind_node]).sum()
            self.YID_dict_tst = self.fn_matrix(dfTrain3, self.F, self.N, isflatten=True)
            print('ID VECTOR DONE', 'dict length is:', len(self.YID_dict_tst.keys()),
                  'dict item shape is:', list(self.YID_dict_tst.values())[0].shape)

            cm.pklsave(os.path.join(self.tmpsavepath, 'tmp_YID_dict_tst.pkl'), self.YID_dict_tst)

            self.YID_dict_tstfn='tmp_YID_dict_tst.pkl'
            self.YID_dict_tst = {}


    def form_feat_matrix(self):
        '''
        Prepares X, Y, YID and Adjacency Matrix from dataset.
        Tuples are formed in alternative way: (atom_index_1. type)

        Returns:
        A dictionary prepared train and test data.
        '''

        if self.dfXtest is None:
            dfTrain1 = self.dfXtrain.copy()
        else:
            dfTrain1 = self.dfXtrain_test.copy()


        atom_ind_lbl='atom_index_1'
        atom_ind_type='ai1_type'

        dfTrain1[atom_ind_type] = list(zip(dfTrain1[atom_ind_lbl], dfTrain1['type']))

        dfTrain1[atom_ind_type] = dfTrain1[atom_ind_type].apply(lambda x: str(x[0]) + '_' + x[1])

        self.F = dfTrain1[atom_ind_type].nunique()
        print('N is', self.N)
        print('F is', self.F)

        self.make_X(dfTrain1)
        self.make_Y(dfTrain1)
        self.make_YID(dfTrain1)


        if self.dfXtest is None:
            champs_data_dict = {'X_dict_trn': self.X_dict,
                                'X_dict_tst': None,
                                'Y_dict_trn': self.Y_dict,
                                'Y_dict_tst': None,
                                'YID_dict_trn': self.YID_dict,
                                'YID_dict_tst': None,
                                'N': self.N,
                                'F': self.F,
                                'label_dim': self.cls_num
                                }
        else:
            champs_data_dict={ 'X_dict_trn': self.X_dict_trn,
                        'X_dict_tst': self.X_dict_tst,
                        'Y_dict_trn': self.Y_dict_trn,
                        'Y_dict_tst': self.Y_dict_tst,
                        'YID_dict_trn': self.YID_dict_trn,
                        'YID_dict_tst': self.YID_dict_tst,
                        'N': self.N,
                        'F':self.F,
                        'label_dim': self.cls_num
                        }

        return champs_data_dict


    @staticmethod
    def fn_matrix(dfTrain2, F, N, isflatten=False):
        '''
        converts a dataframe to a dictionary of 2D nump arrays
        Inputs:
        dfTrain: Dataframe that has two row multi-index levels (molecule name and atom_index_1)
        F: number of features
        N: number of atoms/nodes
        isflatten: enable to make output a flattened vector

        Output:
        A dictionary of numpy ndarrays indexed by molecule name
        '''

        dict_mat_arr = dict()

        lst_ind1 = dfTrain2.index.levels[0].values
        i_ind1 = 0
        len_ind1 = len(lst_ind1)
        for ind1 in lst_ind1:
            matrix_arr = np.zeros((F, N))
            lst_ind2 = dfTrain2.xs(ind1).index.values
            for ind2 in lst_ind2:
                df_feat_cols = dfTrain2.xs((ind1, ind2)).to_numpy()
                matrix_arr[:, ind2] = df_feat_cols
            if isflatten:
                dict_mat_arr[ind1] = np.reshape(matrix_arr, -1)
            else:
                dict_mat_arr[ind1] = matrix_arr
            i_ind1 += 1
            if not (i_ind1 % 5000):
                print(i_ind1, '/', len_ind1, 'molecules are processed')

        return dict_mat_arr

    def make_adjm(self):
        '''
        Prepare weighted adjacency matrix by using coordinate info in structures csv file for each molecule.

        Returns:
        A dictionary containing 2D weighted adjacecny matrix for each molecule in structures csv file.
        '''

        self.dict_adjM = dict()
        grp_dfstructs = self.dfXstructs.groupby('molecule_name')
        i_key = 0
        len_key = len(grp_dfstructs)
        for key, item in grp_dfstructs:
            dfstructs1 = grp_dfstructs.get_group(key)
            dfstructs1 = dfstructs1.set_index('atom_index')
            adjM_temp = np.full((self.N, self.N), self.far_dist)
            for an1 in dfstructs1.index.values:
                for an2 in dfstructs1.index.values:
                    if an1 != an2:
                        dX = (dfstructs1.iloc[an1, 2] - dfstructs1.iloc[an2, 2])
                        dY = (dfstructs1.iloc[an1, 3] - dfstructs1.iloc[an2, 3])
                        dZ = (dfstructs1.iloc[an1, 4] - dfstructs1.iloc[an2, 4])
                        dist = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)
                        if dist > 0:
                            adjM_temp[an1, an2] = 1 / (dist ** self.p)
                        else:
                            print('distance is 1/zero but atoms are different', an1, an2)
                    else:
                        adjM_temp[an1, an2] = self.self_dist
            self.dict_adjM[key] = adjM_temp
            i_key += 1
            if not (i_key % 5000):
                print(i_key, '/', len_key, ' molecules are processed')

        cm.pklsave(os.path.join(self.tmpsavepath, 'tmp_dict_adjM.pkl'), self.dict_adjM)

        self.dict_adjMfn = 'tmp_dict_adjM.pkl'
        self.dict_adjM = {}


    def prep_champs_data(self, gen_file=True):
        '''
        Prepares dataset fully and combines them in one file to be saved.

        Inputs:
        gen_file: enable to generate X, Y, YID, AdjM from raw dataset

        Returns:
        A dictionary keyed with molecule name and has [X, AdjM, Y, YID] in each entry.
        '''
        if gen_file:
            print('Preparing ALT feature matrix, scalar constant vector and ID vector...')
            self.form_feat_matrix()

            print('Preparing ALT adjacency matrix')
            self.make_adjm()

        print('ALL DONE, NOW COMBINING ALT DICTS')

        self.dict_adjM = cm.pklload(os.path.join(self.tmpsavepath, self.dict_adjMfn))

        if self.dfXtest is None:
            self.X_dict=cm.pklload(os.path.join(self.tmpsavepath, self.X_dictfn))
            self.Y_dict=cm.pklload(os.path.join(self.tmpsavepath, self.Y_dictfn))
            self.YID_dict=cm.pklload(os.path.join(self.tmpsavepath, self.YID_dictfn))

            self.comb_dict = dict()
            for key in self.X_dict.keys():
                self.comb_dict[key] = [d[key] for d in [self.X_dict, self.dict_adjM, self.Y_dict, self.YID_dict]]
        else:
            self.comb_dict_trn = dict()
            self.comb_dict_tst = dict()
            print('COMBINING TRAIN DATA DICTS')
            self.X_dict_trn=cm.pklload(os.path.join(self.tmpsavepath, self.X_dict_trnfn))
            self.Y_dict_trn=cm.pklload(os.path.join(self.tmpsavepath, self.Y_dict_trnfn))
            self.YID_dict_trn=cm.pklload(os.path.join(self.tmpsavepath, self.YID_dict_trnfn))
            for key in self.X_dict_trn.keys():
                self.comb_dict_trn[key] = [d[key] for d in [self.X_dict_trn, self.dict_adjM, self.Y_dict_trn, self.YID_dict_trn]]
            print('COMBINING TEST DATA DICTS')
            self.X_dict_tst=cm.pklload(os.path.join(self.tmpsavepath, self.X_dict_tstfn))
            self.Y_dict_tst=cm.pklload(os.path.join(self.tmpsavepath, self.Y_dict_tstfn))
            self.YID_dict_tst=cm.pklload(os.path.join(self.tmpsavepath, self.YID_dict_tstfn))
            for key in self.X_dict_tst.keys():
                self.comb_dict_tst[key] = [d[key] for d in [self.X_dict_tst, self.dict_adjM, self.Y_dict_tst, self.YID_dict_tst]]

        print('DATA IS PREPARED')
        if self.dfXtest is None:
            print('combined dict shape is:', len(self.comb_dict.keys()), [arr.shape for arr in list(self.comb_dict.values())[0]])
            comb_out=[self.comb_dict, None]
        else:
            print('combined train dict shape is:', len(self.comb_dict_trn.keys()), [arr.shape for arr in list(self.comb_dict_trn.values())[0]])
            print('combined test dict shape is:', len(self.comb_dict_tst.keys()), [arr.shape for arr in list(self.comb_dict_tst.values())[0]])
            comb_out=[self.comb_dict_trn, self.comb_dict_tst]

        self.scc_comb_save()

        return comb_out



    def scc_comb_save(self):
        '''
        Saves train and test data in a pickle file
        '''
        self.comb_fn = self.comb_fn if self.comb_fn is not None else 'default_comb_data'
        if self.dfXtest is None:
            cm.pklsave(self.comb_fn+'_raw.pkl', self.comb_dict)
        else:
            cm.pklsave(self.comb_fn+'_trn_raw.pkl', self.comb_dict_trn)
            cm.pklsave(self.comb_fn+'_tst_raw.pkl', self.comb_dict_tst)
        return self.comb_fn


def scc_trnval_slice_save(all_fn, save_fn, trn_end, val_end, tst_end, trn_start=0, val_start=None, tst_start=None, isshuffle=True):
    '''
    Slices the train, val and test data from a preprocessed dataset.

    Inputs:
    all_fn: pickle file that has full data to be sliced
    save_fn: a prefix to save output files
    trn_end: train set end index
    val_end: validation set end index
    tst_end: test set end index
    trn_start: train set start index
    val_start: validation set start index
    tst_start: test set start index
    isshuffle: enable to shuffle full dataset before slicing

    Returns:
    Sliced train, validation and test data as separate files for
    input to graph model and saved in files.
    '''
    train_all = cm.pklload(all_fn)
    keyslst = list(train_all.keys())
    if isshuffle:
        random.shuffle(keyslst)

    trn_X_fm = []
    trn_X_adm = []
    trn_Y_scc = []
    trn_Y_id = []
    val_X_fm = []
    val_X_adm = []
    val_Y_scc = []
    val_Y_id = []
    tst_X_fm = []
    tst_X_adm = []
    tst_Y_scc = []
    tst_Y_id = []


    if trn_start is not None:
        keyslst_trn = keyslst[trn_start:trn_end]
        for key in keyslst_trn:
            trn_X_fm.append(train_all[key][0])
            trn_X_adm.append(train_all[key][1])
            trn_Y_scc.append(train_all[key][2])
            trn_Y_id.append(train_all[key][3])
        cm.pklsave(save_fn+'_X_'+str(trn_end-trn_start)+'_trn.pkl', trn_X_fm)
        cm.pklsave(save_fn + '_Xadjm_'+str(trn_end-trn_start)+'_trn.pkl', trn_X_adm)
        cm.pklsave(save_fn + '_Y_'+str(trn_end-trn_start)+'_trn.pkl', trn_Y_scc)
        cm.pklsave(save_fn + '_Yid_'+str(trn_end-trn_start)+'_trn.pkl', trn_Y_id)

    if val_start is not None:
        keyslst_val = keyslst[val_start:val_end]
        for key in keyslst_val:
            val_X_fm.append(train_all[key][0])
            val_X_adm.append(train_all[key][1])
            val_Y_scc.append(train_all[key][2])
            val_Y_id.append(train_all[key][3])
        cm.pklsave(save_fn+'_X_'+str(val_end-val_start)+'_val.pkl', val_X_fm)
        cm.pklsave(save_fn + '_Xadjm_'+str(val_end-val_start)+'_val.pkl', val_X_adm)
        cm.pklsave(save_fn + '_Y_'+str(val_end-val_start)+'_val.pkl', val_Y_scc)
        cm.pklsave(save_fn + '_Yid_'+str(val_end-val_start)+'_val.pkl', val_Y_id)

    if tst_start is not None:
        keyslst_tst = keyslst[tst_start:tst_end]
        for key in keyslst_tst:
            tst_X_fm.append(train_all[key][0])
            tst_X_adm.append(train_all[key][1])
            tst_Y_scc.append(train_all[key][2])
            tst_Y_id.append(train_all[key][3])
        cm.pklsave(save_fn+'_X_'+str(tst_end-tst_start)+'_test.pkl', tst_X_fm)
        cm.pklsave(save_fn + '_Xadjm_'+str(tst_end-tst_start)+'_test.pkl', tst_X_adm)
        cm.pklsave(save_fn + '_Y_'+str(tst_end-tst_start)+'_test.pkl', tst_Y_scc)
        cm.pklsave(save_fn + '_Yid_'+str(tst_end-tst_start)+'_test.pkl', tst_Y_id)

def scc_test_slice_save(all_fn, save_fn, tst_end=None, tst_start=0, isshuffle=True):
    '''
    Slices only the test dataset from full train dataset

    Inputs:
    all_fn: pickle file that has full data to be sliced
    save_fn: a prefix to save output files
    tst_end: test set end index
    tst_start: test set start index
    isshuffle: enable to shuffle full dataset before slicing

    Returns:
    Sliced test data as separate files for input to graph model and saved in files.
    '''
    test_all = cm.pklload(all_fn)
    keyslst = list(test_all.keys())
    if isshuffle:
        random.shuffle(keyslst)

    tst_X_fm = []
    tst_X_adm = []
    tst_Y_scc = []
    tst_Y_id = []

    if tst_end is None:
        keyslst_tst=keyslst[tst_start:]
        tst_end=len(keyslst_tst)
    else:
        keyslst_tst = keyslst[tst_start:tst_end]

    for key in keyslst_tst:
        tst_X_fm.append(test_all[key][0])
        tst_X_adm.append(test_all[key][1])
        tst_Y_scc.append(test_all[key][2])
        tst_Y_id.append(test_all[key][3])
    cm.pklsave(save_fn + '_X_' + str(tst_end - tst_start) + '_test.pkl', tst_X_fm)
    cm.pklsave(save_fn + '_Xadjm_' + str(tst_end - tst_start) + '_test.pkl', tst_X_adm)
    cm.pklsave(save_fn + '_Yid_' + str(tst_end - tst_start) + '_test.pkl', tst_Y_id)
    cm.pklsave(save_fn + '_Yscc_' + str(tst_end - tst_start) + '_test.pkl', tst_Y_scc)


def scc_load(Xfn, AdjMfn, Yfn=None, Yidfn=None):
    '''
    Wrapper to load input to graph model into memory from pickle file

    Xfn: filename for feature matrix X
    AdjMfn: file name for weighted adjacency matrix
    Yfn: file name for y values
    Yidfn: file name for y label id values.

    Returns:
    Lists of numpy arrays ready as input to training a model.
    '''

    X_lst=cm.pklload(Xfn)
    AdjM_lst=cm.pklload(AdjMfn)

    if Yfn is not None:
        Y_lst=cm.pklload(Yfn)
    else:
        print('Y label data is not given, treating this as test data.')
        Y_lst=None
    if Yidfn is not None:
        Yid_lst=cm.pklload(Yidfn)
    else:
        Yid_lst=None

    return X_lst, AdjM_lst, Y_lst, Yid_lst


def predict2csv(csvfn, y_pred, y_id, y_true=None):
    '''
    Saves the predicted y values with corresponding y label ids in a csv file.

    y_pred: predicted y values
    y_id: y label id values for predicted scalar coupling values
    y_true: real scalar coupling values if available
    csvfn: file name to save as csv

    Returns:
    A dataframe and a csv file that has (y_id, y_pred, [y_true]) columns.
    '''

    y_id_cat=np.concatenate(y_id).astype(int)
    y_pred_cat=np.concatenate(y_pred)

    if y_true is None:
        ydf =  pd.DataFrame(data={'id':y_id_cat, 'scalar_coupling_constant':y_pred_cat})
    else:
        y_true_cat=np.concatenate(y_true)
        ydf = pd.DataFrame(data={'id': y_id_cat, 'pred_scalar_coupling_constant': y_pred_cat, 'true_scalar_coupling_constant':y_true_cat})

    ydf = ydf[ydf['id']>0].sort_values(by='id').reset_index(drop=True)

    ydf.to_csv(csvfn, index=False, header=True)

    print('Data is written in:', csvfn)

    return ydf

