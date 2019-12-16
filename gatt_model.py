'''
CoulGAT: A graph attention model utilizing a screened coulomb potentioal attention.

'''

#import basic packages
import sys
import numpy as np
import os
import tensorflow as tf
import common as cm


class GattModel:
    def __init__(self, param_dict):
        '''
        Initialize graph attention model based on hyperparamaters.
        Inputs:
        param_dict: a dictionary containing hyperparameters for the model.

        Returns:
        a GattModel object.
        '''
        self.N = param_dict['num_nodes']
        self.F = param_dict['num_features']
        self.cls_num = param_dict['class_number']
        self.lay_num = param_dict['num_graph_layers']
        self.blockn = param_dict['resgnn_block_num'] if param_dict['resgnn_block_num'] is not None else 2
        self.F_hid = param_dict['list_hidden_graph_layers']
        self.K_hid = param_dict['list_hidden_heads']
        self.use_Kavg = param_dict['use_head_averaging'] if  param_dict['use_head_averaging'] is not None else False

        self.enable_bn = param_dict['enable_bn'] if param_dict['enable_bn'] is not None else False
        self.bnaxis = 2
        self.bnmomentum = param_dict['bn_momentum'] if param_dict['bn_momentum'] is not None else 0.99

        model_sel_dict={'model_1': self.gmodel,
                        'model_2': self.res_gmodel,
                        'model_3':self.avg_gmodel,
                        'model_4':self.res_avg_gmodel,
                        }
        if param_dict['model_name'] in model_sel_dict.keys():
            self.model_name=param_dict['model_name']
        else:
            print('Indicate the model to run and run again:', model_sel_dict.keys())
            sys.exit(1)

        print('Selected model graph is:', self.model_name)


        if self.model_name == 'model_1':
            if self.use_Kavg:
                self.num_ops=self.lay_num
                self.num_khidden=self.lay_num-1
                print('Doing K avg in last layer')
            else:
                self.num_ops=self.lay_num
                self.num_khidden=self.lay_num

        if self.model_name == 'model_2':
            if self.use_Kavg:
                self.num_ops = self.blockn * (self.lay_num - 1) + 3
                self.num_khidden = self.num_ops - 1
                print('Doing K avg in last layer')
            else:
                self.num_ops = self.blockn * (self.lay_num - 1) + 2
                self.num_khidden = self.num_ops

            self.K_hid=[1]+[self.K_hid]*(self.num_ops-1)
            self.F_hid=[self.F]+[self.F_hid]*(self.num_ops-1)
            print("model_2 K_hid:", self.K_hid)
            print("model_2 F_hid", self.F_hid)

        if self.model_name=='model_3':
            self.K_hid = [1]+self.K_hid*(self.lay_num-1)
            self.F_hid = [self.F]+self.F_hid*(self.lay_num-1)
            print("model_3 K_hid:", self.K_hid)
            print("model_3 F_hid", self.F_hid)


        if self.model_name=='model_4':
            self.K_hid = [1]+self.K_hid*(self.lay_num-1)
            self.F_hid = [self.F]+self.F_hid*(self.lay_num-1)
            print("model_4 K_hid:", self.K_hid)
            print("model_4 F_hid", self.F_hid)

        self.istrain=True

        self.dense_hid = 2*[self.K_hid[-1]*self.F_hid[-1]] + [self.cls_num]

        print('dense hid layers are:', self.dense_hid)

        self.batch_size = param_dict['batch_size'] if param_dict['batch_size'] is not None else 50
        self.num_epochs = param_dict['num_epochs'] if param_dict['num_epochs'] is not None else 100
        self.lr = param_dict['learning_rate'] if param_dict['learning_rate'] is not None else 0.001
        self.reg_scale= param_dict['reg_scale'] if param_dict['reg_scale'] is not None else 0.01
        self.loss_type= param_dict['loss_type'] if param_dict['loss_type'] is not None else 'SCCLMAE'
        self.trn_in_keep_prob = param_dict['trn_in_keep_prob'] if param_dict['trn_in_keep_prob'] is not None else 1.0
        self.trn_eij_keep_prob = param_dict['trn_eij_keep_prob'] if param_dict['trn_eij_keep_prob'] is not None else 1.0
        self.enable_pw=param_dict['enable_pw'] if param_dict['enable_pw'] is not None else False
        self.is_classify = param_dict['is_classify'] if param_dict['is_classify'] is not None else False

        self.num_early_stop = param_dict['num_early_stop'] if param_dict['num_early_stop'] is not None else 0.5
        self.early_stop_int = int(np.ceil((self.num_epochs-1) / self.num_early_stop))
        self.early_stop_threshold = param_dict['early_stop_threshold'] if param_dict['early_stop_threshold'] is not None else 0

        self.models_folder=param_dict['models_folder'] if param_dict['models_folder'] is not None else 'tmp_saved_models'
        self.sum_folder=param_dict['sum_folder'] if param_dict['sum_folder'] is not None else 'summaries'
        self.postfix=param_dict['label']+'_'+self.model_name if param_dict['label'] is not None else self.model_name

        self.batch_count = None

        self.gat_graph=tf.Graph()

        model_param_dict=model_sel_dict[self.model_name](self.gat_graph)

        self.init=model_param_dict['init']
        self.train_op=model_param_dict['train_op']
        self.loss=model_param_dict['loss']
        self.y_pred=model_param_dict['y_pred']
        self.pwlst_hid=model_param_dict['adjm_power']
        self.alst_hid=model_param_dict['alst_hid']
        self.Wlst_hid=model_param_dict['Wlst_hid']
        self.blst_hid=model_param_dict['blst_hid']
        self.Hlst=model_param_dict['Hlst']
        self.dense_layers=model_param_dict['dense_layers']
        self.X=model_param_dict['X']
        self.AdjM=model_param_dict['AdjM']
        self.y=model_param_dict['y']
        self.X_in_var=model_param_dict['X_in_var']
        self.Xadjm_in_var=model_param_dict['Xadjm_in_var']
        self.Y_in_var=model_param_dict['Y_in_var']
        self.YID_in_var = model_param_dict['YID_in_var']
        self.sbuff_size=model_param_dict['shuffle_buff_size']
        self.val_X_in_var=model_param_dict['val_X_in_var']
        self.val_Xadjm_in_var=model_param_dict['val_Xadjm_in_var']
        self.val_Y_in_var=model_param_dict['val_Y_in_var']
        self.val_YID_in_var=model_param_dict['val_YID_in_var']
        self.saver=model_param_dict['saver']
        self.trn_iterator=model_param_dict['trn_iterator']
        self.val_iterator=model_param_dict['val_iterator']

        self.loss_early_stop = 1000

        print('parameters are initialized from dict:', param_dict)
        cm.pklsave('saved_model_params/hyperparams_'+self.postfix+'.pkl', param_dict)

    @staticmethod
    def single_attn_mod( a, W, b, H, Adj, act_att=tf.nn.leaky_relu, pw=None):
        '''
        Calculates the attention coefficient matrix E

        Inputs:
        a: learnable attention vector.
        W: Weight tensor
        b: bias tensor
        H: input tensor to hidden layer
        Adj: weighted adjacency matrix
        act_att: activation function used in the attention

        Returns:
        E:  the softmax normalized attention function output
        '''

        We = tf.tile(tf.expand_dims(W, 0), tf.stack([tf.shape(H)[0], 1, 1]))
        WH = tf.add(tf.matmul(We, H), b)  # dim batch_size, F' N

        if pw is not None:
            pwe = tf.tile(tf.expand_dims(pw, 0), tf.stack([tf.shape(H)[0], 1, 1]))
            Adj=tf.math.pow(Adj, pwe, name="Adj_pw")

        Adj = tf.nn.softmax(Adj, axis=-1)

        ae = tf.tile(tf.expand_dims(a, 0), tf.stack([tf.shape(H)[0], 1, 1]))
        WHiWHj = tf.matmul(WH, tf.transpose(Adj, [0, 2, 1]))
        E = act_att(tf.matmul(ae, WHiWHj))
        En = tf.nn.softmax(E, axis=-1)

        return En, WH


    @staticmethod
    def single_hidden_out(E, WH, act_hidden=tf.nn.relu, in_keep_prob=1.0, eij_keep_prob=1.0):
        '''
        Calculates the hidden layer output for single head.

        Inputs:
        E: attention coefficient matrix
        WH: Matrix multiplication of Weight tensor and input tensor
        act_hidden: activation function for layer output
        in_keep_prob:1 - drop out rate for WH
        eij_keep_prob:1- drop out rate for E

        Returns:
        Next hidden layer tensor for single head
        '''

        if in_keep_prob < 1.0:
            WH = tf.nn.dropout(WH, rate=1 - in_keep_prob)

        if eij_keep_prob < 1.0:
            E = tf.nn.dropout(E, rate=1 - eij_keep_prob)

        H_next = act_hidden(tf.matmul(WH, tf.transpose(E, [0, 2,1])))

        return H_next

    def K_hidden_out(self, H, A, alst, Wlst, blst, K=1, clsfy_layer=False, in_keep_prob=1.0,
                     eij_keep_prob=1.0, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=None):
        '''
        Calculates the hidden layer output for K heads

        Inputs:
        H: input tensor
        A: weighted adjacency matrix
        alst: List of attention vectors for the hidden layer
        Wlst: list of weight tensors for the hidden layer
        K: number of heads
        clsfy_layer: True: pooling, False: concatenating
        in_keep_prob:1 - drop out rate for WH
        eij_keep_prob:1- drop out rate for E
        act_hidden: activation function for layer output
        act_att: activation function used in the attention
        pwlist: list of learnable power matrix

        Returns:
        Next hidden layer tensor
        '''
        Hlst = []
        if not clsfy_layer:
            act_input=act_hidden
        else:
            act_input=tf.identity

        for i in range(K):
            pwlst_item=pwlst[i] if pwlst is not None else None
            Ei, WlstiH = self.single_attn_mod(alst[i], Wlst[i], blst[i], H, A, act_att=act_att, pw=pwlst_item)

            Hi = self.single_hidden_out(Ei, WlstiH, act_hidden=act_input,
                                   in_keep_prob=in_keep_prob, eij_keep_prob=eij_keep_prob)
            Hlst.append(Hi)

        if not clsfy_layer:
            H_next = tf.concat(Hlst, axis=1)
        else:
            H_next = tf.add_n(Hlst) / K

        return H_next

    def res_K_hidden_out(self, H, A, alst, Wlst, blst, K, clsfy_layer=False, in_keep_prob=1.0,
                     eij_keep_prob=1.0, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=None, en_BN=False):
        '''
        Calculates the hidden layer output for K heads for residual networks.

        Inputs:
        H: input tensor
        A: weighted adjacency matrix
        alst: List of attention vectors for the hidden layer
        Wlst: list of weight tensors for the hidden layer
        K: number of heads
        clsfy_layer: True: pooling, False: concatenating
        in_keep_prob:1 - drop out rate for WH
        eij_keep_prob:1- drop out rate for E
        act_hidden: activation function for layer output
        act_att: activation function used in the attention
        pwlist: list of learnable power matrix
        en_BN: enable abtch normalization before activation

        Returns:
        List of hidden layer tensors

        '''

        Hresb=[H]

        for i in range(0, self.blockn):
            if en_BN:
                Hresb[i]=tf.layers.batch_normalization(Hresb[i], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
            Hresb.append(self.K_hidden_out(act_hidden(Hresb[i]), A, alst[i], Wlst[i], blst[i], K=K[i], clsfy_layer=clsfy_layer, in_keep_prob=in_keep_prob,
                     eij_keep_prob=eij_keep_prob, act_att=act_att, act_hidden=tf.identity, pwlst=pwlst[i]))

        Hresb[-1]=Hresb[-1]+H

        return Hresb[1:]

    def avg_res_K_hidden_out(self, H, A, alst, Wlst, blst, K, clsfy_layer=False, in_keep_prob=1.0,
                     eij_keep_prob=1.0, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=None, en_BN=False, en_res=True):
        '''
        Calculates the hidden layer output for K heads for residual networks with averaging.

        Inputs:
        H: input tensor
        A: weighted adjacency matrix
        alst: List of attention vectors for the hidden layer
        Wlst: list of weight tensors for the hidden layer
        K: number of heads
        clsfy_layer: True: pooling, False: concatenating
        in_keep_prob:1 - drop out rate for WH
        eij_keep_prob:1- drop out rate for E
        act_hidden: activation function for layer output
        act_att: activation function used in the attention
        pwlist: list of learnable power matrix
        en_BN: enable abtch normalization before activation
        en_res: Enable residual connection

        Returns:
        List of hidden layer tensors calculated within resblock
        '''

        Hresb=[H]

        for i in range(0, self.blockn-1):
            pwlst_item = pwlst[i] if pwlst is not None else None
            if en_BN:
                Hresb[i]=tf.layers.batch_normalization(Hresb[i], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
            Hresb.append(self.K_hidden_out(act_hidden(Hresb[i]), A, alst[i], Wlst[i], blst[i], K=K[i], clsfy_layer=clsfy_layer, in_keep_prob=in_keep_prob,
                     eij_keep_prob=eij_keep_prob, act_att=act_att, act_hidden=tf.identity, pwlst=pwlst_item))
        if en_BN:
            Hresb[-1]=tf.layers.batch_normalization(Hresb[-1], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
        pwlst_item = pwlst[-1] if pwlst is not None else None
        Hresb.append(self.K_hidden_out(act_hidden(Hresb[-1]), A, alst[-1], Wlst[-1], blst[-1], K=K[-1],
                                       clsfy_layer=True, in_keep_prob=in_keep_prob,
                                       eij_keep_prob=eij_keep_prob, act_att=act_att, act_hidden=tf.identity,
                                       pwlst=pwlst_item))

        if en_res:
            Hresb[-1]=Hresb[-1]+H

        return Hresb[1:]

    def avg_K_hidden_out(self, H, A, alst, Wlst, blst, K, clsfy_layer=False, in_keep_prob=1.0,
                     eij_keep_prob=1.0, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=None, en_BN=False, en_fin_act=True):
        '''
        Calculates the hidden layer output for K heads for plain network with averaging.

        Inputs:
        H: input tensor
        A: weighted adjacency matrix
        alst: List of attention vectors for the hidden layer
        Wlst: list of weight tensors for the hidden layer
        K: number of heads
        clsfy_layer: True: pooling, False: concatenating
        in_keep_prob:1 - drop out rate for WH
        eij_keep_prob:1- drop out rate for E
        act_hidden: activation function for layer output
        act_att: activation function used in the attention
        pwlist: list of learnable power matrix
        en_BN: enable abtch normalization before activation
        en_fin_act: Apply activation to output if true.

        Returns:
        List of hidden layer tensors calculated within resblock

        '''
        Hresb=[H]

        for i in range(0, self.blockn-1):
            pwlst_item = pwlst[i] if pwlst is not None else None
            Hresb.append(self.K_hidden_out(Hresb[i], A, alst[i], Wlst[i], blst[i], K=K[i], clsfy_layer=clsfy_layer, in_keep_prob=in_keep_prob,
                     eij_keep_prob=eij_keep_prob, act_att=act_att, act_hidden=tf.identity, pwlst=pwlst_item))
            if en_BN:
                Hresb[-1]=tf.layers.batch_normalization(Hresb[-1], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
            Hresb[-1]=act_hidden(Hresb[-1])

        pwlst_item = pwlst[-1] if pwlst is not None else None
        Hresb.append(self.K_hidden_out(Hresb[-1], A, alst[-1], Wlst[-1], blst[-1], K=K[-1],
                                       clsfy_layer=True, in_keep_prob=in_keep_prob,
                                       eij_keep_prob=eij_keep_prob, act_att=act_att, act_hidden=tf.identity,
                                       pwlst=pwlst_item))
        if en_BN:
            Hresb[-1]=tf.layers.batch_normalization(Hresb[-1], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
        if en_fin_act:
            Hresb[-1]=act_hidden(Hresb[-1])

        return Hresb[1:]



    def make_iter(self):
        '''
        Generate reinitializable iterators for training and validation.
        '''

        X_in_var = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="X_in_var")
        Xadjm_in_var = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="Xadjm_in_var")
        Y_in_var = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="Y_in_var")
        YID_in_var = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="YID_in_var")
        sbuff_size=tf.placeholder(tf.int64, shape=[], name="shuffle_buff_size")

        val_X_in_var = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="val_X_in_var")
        val_Xadjm_in_var = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="val_Xadjm_in_var")
        val_Y_in_var = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="val_Y_in_var")
        val_YID_in_var = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="val_YID_in_var")


        dataset1ka = tf.data.Dataset.from_tensor_slices(X_in_var)
        dataset1kb = tf.data.Dataset.from_tensor_slices(Xadjm_in_var)
        dataset1kc = tf.data.Dataset.from_tensor_slices(Y_in_var)
        dataset1kd = tf.data.Dataset.from_tensor_slices(YID_in_var)
        dataset1k = tf.data.Dataset.zip((dataset1ka, dataset1kb, dataset1kc, dataset1kd))
        print("Train Dataset output types:", dataset1k.output_types)
        print("Train Dataset output shapes:", dataset1k.output_shapes)

        dataset1k = dataset1k.shuffle(buffer_size=sbuff_size)
        dataset1k = dataset1k.repeat()
        batch_dataset1k = dataset1k.batch(self.batch_size).prefetch(1)

        val_dataset1ka = tf.data.Dataset.from_tensor_slices(val_X_in_var)
        val_dataset1kb = tf.data.Dataset.from_tensor_slices(val_Xadjm_in_var)
        val_dataset1kc = tf.data.Dataset.from_tensor_slices(val_Y_in_var)
        val_dataset1kd = tf.data.Dataset.from_tensor_slices(val_YID_in_var)
        val_dataset1k = tf.data.Dataset.zip((val_dataset1ka, val_dataset1kb, val_dataset1kc, val_dataset1kd))
        print("Validation Dataset output types:", val_dataset1k.output_types)
        print("Validation Dataset output shapes:", val_dataset1k.output_shapes)

        val_batch_dataset1k = val_dataset1k.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(batch_dataset1k.output_types, batch_dataset1k.output_shapes)

        trn_iterator=iterator.make_initializer(batch_dataset1k)
        val_iterator=iterator.make_initializer(val_batch_dataset1k)

        data_param_dict={
            'iterator':iterator,
            'trn_iterator':trn_iterator,
            'val_iterator':val_iterator,
            'X_in_var': X_in_var,
            'Xadjm_in_var': Xadjm_in_var,
            'Y_in_var': Y_in_var,
            'YID_in_var': YID_in_var,
            'val_X_in_var': val_X_in_var,
            'val_Xadjm_in_var': val_Xadjm_in_var,
            'val_Y_in_var': val_Y_in_var,
            'val_YID_in_var': val_YID_in_var,
            'shuffle_buff_size':sbuff_size
        }

        return data_param_dict

    def scclmae(self, y, y_pred, yid):
        '''
        Calculates the SCCLMAE score

        Inputs:
        y: true values
        y_pred: predicted values
        yid: id values for nor-zero entries of y

        Returns:
        SCCLMAE loss
        '''

        yid_nz_cond=tf.greater(yid, 0)
        yid_nz_ind=tf.where_v2(yid_nz_cond)

        y_nz=tf.gather_nd(y, yid_nz_ind)
        y_pred_nz=tf.gather_nd(y_pred, yid_nz_ind)

        lmae_loss=tf.math.log(1e-9+tf.losses.absolute_difference(y_nz, y_pred_nz))

        return lmae_loss

    # model_1
    def gmodel(self, gat_graph_in):
        '''
        Model_1: Plain graph attention network with/without pooling as final layer.
        '''
        reset_graph()

        with gat_graph_in.as_default():
            set_graph_seed()

            X = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="X_inp")
            AdjM = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="adja")
            y = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="y_label")
            yid = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="y_label_id")

            data_param_dict_in = self.make_iter()

            X, AdjM, y, yid = data_param_dict_in['iterator'].get_next()

            alst_hid = []
            Wlst_hid = []
            blst_hid = []
            pwlst_hid = []

            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg_scale)
            pw_initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("gat_a_and_W", reuse=tf.AUTO_REUSE):
                for h in range(1, self.num_ops):
                    alst_hid.append([tf.get_variable("alst_hid" + str(h) + str(i), shape=(self.N, self.F_hid[h]),
                                                     dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer()) for i in
                                     range(self.K_hid[h])])

                    if self.enable_pw is True:
                        pwlst_hid.append([tf.get_variable("pwlst_hid" + str(h) + str(i), shape=(self.N, self.N),
                                                          dtype=tf.float32,
                                                          initializer=pw_initializer) for i in range(self.K_hid[h])])

                    Wlst_hid.append([tf.get_variable("Wlst_hid" + str(h) + str(i),
                                                     shape=(self.F_hid[h], self.K_hid[h - 1] * self.F_hid[h - 1]),
                                                     dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer()) for i in
                                     range(self.K_hid[h])])

                    blst_hid.append([tf.get_variable("blst_hid" + str(h) + str(i), shape=(self.F_hid[h], self.N),
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.zeros()) for i in
                                     range(self.K_hid[h])])

            Hlst = []
            Hlst.append(X)

            for h in range(1, self.num_khidden):
                pwlst_item = pwlst_hid[h - 1] if self.enable_pw is True else None

                Hlst.append(self.K_hidden_out(Hlst[h - 1], AdjM, alst_hid[h - 1], Wlst_hid[h - 1], blst_hid[h - 1],
                                              K=self.K_hid[h],
                                              clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                              eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu,
                                              act_hidden=tf.nn.relu, pwlst=pwlst_item))

            if self.use_Kavg:
                pwlst_item = pwlst_hid[h - 1] if self.enable_pw is True else None
                Hlst.append(
                    self.K_hidden_out(Hlst[-1], AdjM, alst_hid[-1], Wlst_hid[-1], blst_hid[-1], K=self.K_hid[-1],
                                      clsfy_layer=True, in_keep_prob=self.trn_in_keep_prob,
                                      eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu,
                                      act_hidden=tf.nn.relu,
                                      pwlst=pwlst_item))
                Hlst[-1] = tf.nn.relu(Hlst[-1])

            flat_H = tf.reshape(Hlst[-1], (-1, Hlst[-1].shape[1] * Hlst[-1].shape[2]))
            print('Input shape is:', X.shape)
            print('Last output shape is:', Hlst[-1].shape)
            print('Flattened last output shape is:', flat_H.shape)

            if len(self.dense_hid) > 1:
                dense_layers = [tf.layers.dense(inputs=flat_H, units=self.dense_hid[0], activation=tf.nn.relu,
                                                kernel_initializer='glorot_uniform', use_bias=True,
                                                bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

                for n in range(1, len(self.dense_hid[:-1])):
                    dense_layers.append(
                        tf.layers.dense(inputs=dense_layers[n - 1], units=self.dense_hid[n], activation=tf.nn.relu,
                                        kernel_initializer='glorot_uniform', use_bias=True,
                                        bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

                dense_layers.append(
                    tf.layers.dense(inputs=dense_layers[-1], units=self.dense_hid[-1], activation=None,
                                    kernel_initializer='glorot_uniform',
                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

            else:
                dense_layers = [tf.layers.dense(inputs=flat_H, units=self.dense_hid[-1], activation=None,
                                                kernel_initializer='glorot_uniform',
                                                use_bias=True, bias_initializer='zeros',
                                                kernel_regularizer=kernel_regularizer)]

            y_pred = dense_layers[-1]

            loss_dict = {
                'MSE': tf.losses.mean_squared_error(y, y_pred),
                'MAE': tf.losses.absolute_difference(y, y_pred),
                'HUBER': tf.losses.huber_loss(y, y_pred, delta=1.0),
                'SCCLMAE': self.scclmae(y, y_pred, yid)
            }

            if self.is_classify:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred)
            else:
                loss = loss_dict[self.loss_type]

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([loss] + reg_losses)

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            train_op = opt.minimize(loss)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        model_param_dict = {
            'X': X,
            'AdjM': AdjM,
            'y': y,
            'init': init,
            'train_op': train_op,
            'loss': loss,
            'y_pred': y_pred,
            'adjm_power': pwlst_hid,
            'alst_hid': alst_hid,
            'Wlst_hid': Wlst_hid,
            'blst_hid': blst_hid,
            'Hlst': Hlst,
            'dense_layers': dense_layers,
            'saver': saver,
            'X_in_var': data_param_dict_in['X_in_var'],
            'Xadjm_in_var': data_param_dict_in['Xadjm_in_var'],
            'Y_in_var': data_param_dict_in['Y_in_var'],
            'YID_in_var': data_param_dict_in['YID_in_var'],
            'val_X_in_var': data_param_dict_in['val_X_in_var'],
            'val_Xadjm_in_var': data_param_dict_in['val_Xadjm_in_var'],
            'val_Y_in_var': data_param_dict_in['val_Y_in_var'],
            'val_YID_in_var': data_param_dict_in['val_YID_in_var'],
            'shuffle_buff_size': data_param_dict_in['shuffle_buff_size'],
            'trn_iterator': data_param_dict_in['trn_iterator'],
            'val_iterator': data_param_dict_in['val_iterator']
        }

        return model_param_dict

    #model_2
    def res_gmodel(self, gat_graph_in):
        '''
        Model_2: Residual graph attention network with/without pooling as final layer.
        '''

        reset_graph()

        with gat_graph_in.as_default():

            set_graph_seed()

            X = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="X_inp")
            AdjM = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="adja")
            y = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="y_label")
            yid = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="y_label_id")

            data_param_dict_in=self.make_iter()
            X, AdjM, y, yid = data_param_dict_in['iterator'].get_next()

            alst_hid = []
            Wlst_hid = []
            blst_hid = []
            pwlst_hid=[]


            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale)

            pw_initializer=tf.contrib.layers.xavier_initializer()

            bn=self.blockn

            with tf.variable_scope("gat_a_and_W", reuse=tf.AUTO_REUSE):
                for h in range(1, self.num_ops):
                    alst_hid.append([tf.get_variable("alst_hid" + str(h) + str(i), shape=(self.N, self.F_hid[h]), dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer()) for i in
                                     range(self.K_hid[h])])

                    if self.enable_pw is True:
                        pwlst_hid.append([tf.get_variable("pwlst_hid" + str(h) + str(i), shape=(self.N, self.N), dtype=tf.float32,
                                                     initializer=pw_initializer) for i in range(self.K_hid[h])])

                    Wlst_hid.append([tf.get_variable("Wlst_hid" + str(h) + str(i),
                                                     shape=(self.F_hid[h], self.K_hid[h - 1] * self.F_hid[h - 1]), dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer()) for i in
                                     range(self.K_hid[h])])

                    blst_hid.append([tf.get_variable("blst_hid" + str(h) + str(i), shape=(self.F_hid[h], self.N), dtype=tf.float32,
                                                     initializer=tf.initializers.zeros()) for i in range(self.K_hid[h])])

            Hlst = []
            Hlst.append(X)

            pwlst_item = pwlst_hid[0] if self.enable_pw is True else None
            Hlst.append(self.K_hidden_out(Hlst[0], AdjM, alst_hid[0], Wlst_hid[0], blst_hid[0], K=self.K_hid[1],
                                         clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                         eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.identity, pwlst=pwlst_item))

            if self.enable_bn:
                Hlst[-1] = tf.layers.batch_normalization(Hlst[-1], axis=self.bnaxis, training=self.istrain, momentum=self.bnmomentum)
            Hlst[-1] = tf.nn.relu(Hlst[-1])

            for h in range(2, self.num_khidden, bn):
                pwlst_item = pwlst_hid[h-1:h-1+bn] if self.enable_pw is True else None

                Hlst=Hlst+ self.res_K_hidden_out(Hlst[h-1], AdjM, alst_hid[h-1:h-1+bn], Wlst_hid[h-1:h-1+bn], blst_hid[h-1:h-1+bn], K=self.K_hid[h:h+bn],
                                         clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                         eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=pwlst_item, en_BN=self.enable_bn)

            Hlst[-1] = tf.nn.relu(Hlst[-1])

            if self.use_Kavg:
                Hlst.append(self.K_hidden_out(Hlst[-1], AdjM, alst_hid[-1], Wlst_hid[-1], blst_hid[-1], K=self.K_hid[-1],
                                      clsfy_layer=True, in_keep_prob=self.trn_in_keep_prob,
                                      eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu,
                                      pwlst=pwlst_hid[-1]))
                Hlst[-1] = tf.nn.relu(Hlst[-1])

            flat_H = tf.reshape(Hlst[-1], (-1, Hlst[-1].shape[1] * Hlst[-1].shape[2]))
            print('Input shape is:', X.shape)
            print('Last output shape is:', Hlst[-1].shape)
            print('Flattened last output shape is:', flat_H.shape)

            if len(self.dense_hid) > 1:
                dense_layers = [tf.layers.dense(inputs=flat_H, units=self.dense_hid[0], activation=tf.nn.relu,
                                                kernel_initializer='glorot_uniform', use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

                for n in range(1, len(self.dense_hid[:-1])):
                    dense_layers.append(tf.layers.dense(inputs=dense_layers[n - 1], units=self.dense_hid[n], activation=tf.nn.relu,
                                                     kernel_initializer='glorot_uniform', use_bias=True,
                                                     bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

                dense_layers.append(tf.layers.dense(inputs=dense_layers[-1], units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

            else:
                dense_layers= [tf.layers.dense(inputs=flat_H, units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

            y_pred = dense_layers[-1]

            loss_dict={
                'MSE': tf.losses.mean_squared_error(y, y_pred),
                'MAE': tf.losses.absolute_difference(y, y_pred),
                'HUBER': tf.losses.huber_loss(y, y_pred, delta=1.0),
                'SCCLMAE': self.scclmae(y, y_pred, yid)
            }

            if self.is_classify:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred)
            else:
                loss = loss_dict[self.loss_type]

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss=tf.add_n([loss]+reg_losses)

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            train_op = opt.minimize(loss)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        model_param_dict={
            'X': X,
            'AdjM': AdjM,
            'y': y,
            'init': init,
            'train_op':train_op,
            'loss':loss,
            'y_pred': y_pred,
            'adjm_power':pwlst_hid,
            'alst_hid':alst_hid,
            'Wlst_hid':Wlst_hid,
            'blst_hid': blst_hid,
            'Hlst':Hlst,
            'dense_layers': dense_layers,
            'saver': saver,
            'X_in_var':data_param_dict_in['X_in_var'],
            'Xadjm_in_var': data_param_dict_in['Xadjm_in_var'],
            'Y_in_var': data_param_dict_in['Y_in_var'],
            'YID_in_var': data_param_dict_in['YID_in_var'],
            'val_X_in_var': data_param_dict_in['val_X_in_var'],
            'val_Xadjm_in_var': data_param_dict_in['val_Xadjm_in_var'],
            'val_Y_in_var': data_param_dict_in['val_Y_in_var'],
            'val_YID_in_var': data_param_dict_in['val_YID_in_var'],
            'shuffle_buff_size':data_param_dict_in['shuffle_buff_size'],
            'trn_iterator': data_param_dict_in['trn_iterator'],
            'val_iterator': data_param_dict_in['val_iterator']
        }

        return model_param_dict


    #model_3
    def avg_gmodel(self, gat_graph_in):
        '''
        model_3: plain graph attention network formed by attention layer blocks
        with pooling as last layer.
        '''

        reset_graph()

        with gat_graph_in.as_default():
            set_graph_seed()

            X = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="X_inp")
            AdjM = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="adja")
            y = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="y_label")
            yid = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="y_label_id")

            data_param_dict_in=self.make_iter()
            X, AdjM, y, yid = data_param_dict_in['iterator'].get_next()

            alst_hid = []
            Wlst_hid = []
            blst_hid = []
            pwlst_hid=[]

            bn=self.blockn

            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale)
            pw_initializer=tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("gat_a_and_W", reuse=tf.AUTO_REUSE):
                for h in range(1, (self.lay_num-1)*bn+1, bn):
                    for k in range(0, bn):
                        if k==0:
                            K_mult=1
                        else:
                            K_mult=self.K_hid[h+k - 1]

                        alst_hid.append([tf.get_variable("alst_hid" + str(h) +str(k)+str(i), shape=(self.N, self.F_hid[h]), dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer()) for i in
                                         range(self.K_hid[h+k])])

                        if self.enable_pw is True:
                            pwlst_hid.append([tf.get_variable("pwlst_hid" + str(h)+str(k)+str(i), shape=(self.N, self.N), dtype=tf.float32,
                                                         initializer=pw_initializer) for i in range(self.K_hid[h+k])])


                        Wlst_hid.append([tf.get_variable("Wlst_hid" + str(h)+str(k)+str(i),
                                                         shape=(self.F_hid[h+k], K_mult * self.F_hid[h+k - 1]), dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer()) for i in
                                         range(self.K_hid[h+k])])

                        blst_hid.append([tf.get_variable("blst_hid" + str(h) +str(k)+ str(i), shape=(self.F_hid[h+k], self.N), dtype=tf.float32,
                                                         initializer=tf.initializers.zeros()) for i in range(self.K_hid[h+k])])

            Hlst = []
            Hlst.append(X)

            for h in range(1, (self.lay_num-1)*bn+1, bn):
                pwlst_item=pwlst_hid[h-1:h-1+bn] if self.enable_pw is True else None
                Hlst=Hlst + self.avg_K_hidden_out(Hlst[h-1], AdjM, alst_hid[h-1:h-1+bn], Wlst_hid[h-1:h-1+bn], blst_hid[h-1:h-1+bn], self.K_hid[h:h+bn], clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                     eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=pwlst_item,
                                     en_BN=self.enable_bn, en_fin_act=True)

            flat_H = tf.reshape(Hlst[-1], (-1, Hlst[-1].shape[1] * Hlst[-1].shape[2]))
            print('Input shape is:', X.shape)
            print('Last output shape is:', Hlst[-1].shape)
            print('Flattened last output shape is:', flat_H.shape)

            if len(self.dense_hid) > 1:
                dense_layers = [tf.layers.dense(inputs=flat_H, units=self.dense_hid[0], activation=tf.nn.relu,
                                                kernel_initializer='glorot_uniform', use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

                for n in range(1, len(self.dense_hid[:-1])):
                    dense_layers.append(tf.layers.dense(inputs=dense_layers[n - 1], units=self.dense_hid[n], activation=tf.nn.relu,
                                                     kernel_initializer='glorot_uniform', use_bias=True,
                                                     bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

                dense_layers.append(tf.layers.dense(inputs=dense_layers[-1], units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

            else:
                dense_layers= [tf.layers.dense(inputs=flat_H, units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

            y_pred = dense_layers[-1]

            loss_dict={
                'MSE': tf.losses.mean_squared_error(y, y_pred),
                'MAE': tf.losses.absolute_difference(y, y_pred),
                'HUBER': tf.losses.huber_loss(y, y_pred, delta=1.0),
                'SCCLMAE': self.scclmae(y, y_pred, yid)
            }

            if self.is_classify:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred)
            else:
                loss = loss_dict[self.loss_type]

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss=tf.add_n([loss]+reg_losses)

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            train_op = opt.minimize(loss)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        model_param_dict={
            'X': X,
            'AdjM': AdjM,
            'y': y,
            'init': init,
            'train_op':train_op,
            'loss':loss,
            'y_pred': y_pred,
            'adjm_power':pwlst_hid,
            'alst_hid':alst_hid,
            'Wlst_hid':Wlst_hid,
            'blst_hid': blst_hid,
            'Hlst':Hlst,
            'dense_layers': dense_layers,
            'saver': saver,
            'X_in_var':data_param_dict_in['X_in_var'],
            'Xadjm_in_var': data_param_dict_in['Xadjm_in_var'],
            'Y_in_var': data_param_dict_in['Y_in_var'],
            'YID_in_var': data_param_dict_in['YID_in_var'],
            'val_X_in_var': data_param_dict_in['val_X_in_var'],
            'val_Xadjm_in_var': data_param_dict_in['val_Xadjm_in_var'],
            'val_Y_in_var': data_param_dict_in['val_Y_in_var'],
            'val_YID_in_var': data_param_dict_in['val_YID_in_var'],
            'shuffle_buff_size':data_param_dict_in['shuffle_buff_size'],
            'trn_iterator': data_param_dict_in['trn_iterator'],
            'val_iterator': data_param_dict_in['val_iterator']
        }

        return model_param_dict

    #model_4
    def res_avg_gmodel(self, gat_graph_in):
        '''
        model_4: Residual graph attention network formed by res-blocks
        that has pooling in the last attention layer.
        '''
        reset_graph()

        with gat_graph_in.as_default():
            set_graph_seed()

            X = tf.placeholder(tf.float32, shape=[None, self.F, self.N], name="X_inp")
            AdjM = tf.placeholder(tf.float32, shape=[None, self.N, self.N], name="adja")
            y = tf.placeholder(tf.float32, shape=[None, self.cls_num], name="y_label")
            yid = tf.placeholder(tf.int32, shape=[None, self.cls_num], name="y_label_id")

            data_param_dict_in=self.make_iter()
            X, AdjM, y, yid = data_param_dict_in['iterator'].get_next()

            alst_hid = []
            Wlst_hid = []
            blst_hid = []
            pwlst_hid=[]

            bn=self.blockn

            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_scale)
            pw_initializer=tf.contrib.layers.xavier_initializer()

            with tf.variable_scope("gat_a_and_W", reuse=tf.AUTO_REUSE):
                for h in range(1, (self.lay_num-1)*bn+1, bn):
                    for k in range(0, bn):
                        if k==0:
                            K_mult=1
                        else:
                            K_mult=self.K_hid[h+k - 1]

                        alst_hid.append([tf.get_variable("alst_hid" + str(h) +str(k)+str(i), shape=(self.N, self.F_hid[h]), dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer()) for i in
                                         range(self.K_hid[h+k])])

                        if self.enable_pw is True:
                            pwlst_hid.append([tf.get_variable("pwlst_hid" + str(h)+str(k)+str(i), shape=(self.N, self.N), dtype=tf.float32,
                                                         initializer=pw_initializer) for i in range(self.K_hid[h+k])])

                        Wlst_hid.append([tf.get_variable("Wlst_hid" + str(h)+str(k)+str(i),
                                                         shape=(self.F_hid[h+k], K_mult * self.F_hid[h+k - 1]), dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer()) for i in
                                         range(self.K_hid[h+k])])

                        blst_hid.append([tf.get_variable("blst_hid" + str(h) +str(k)+ str(i), shape=(self.F_hid[h+k], self.N), dtype=tf.float32,
                                                         initializer=tf.initializers.zeros()) for i in range(self.K_hid[h+k])])

            Hlst = []
            Hlst.append(X)

            pwlst_item = pwlst_hid[0:bn] if self.enable_pw is True else None
            Hlst=Hlst + self.avg_K_hidden_out(Hlst[0], AdjM, alst_hid[0:bn], Wlst_hid[0:bn], blst_hid[0:bn],
                                                      self.K_hid[1:bn+1], clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                     eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=pwlst_item,
                                     en_BN=self.enable_bn, en_fin_act=True)

            for h in range(bn+1, (self.lay_num-1)*bn+1, bn):
                pwlst_item = pwlst_hid[h-1:h-1+bn] if self.enable_pw is True else None
                Hlst=Hlst + self.avg_res_K_hidden_out(Hlst[h-1], AdjM, alst_hid[h-1:h-1+bn], Wlst_hid[h-1:h-1+bn], blst_hid[h-1:h-1+bn],
                                                      self.K_hid[h:h+bn], clsfy_layer=False, in_keep_prob=self.trn_in_keep_prob,
                                     eij_keep_prob=self.trn_eij_keep_prob, act_att=tf.nn.leaky_relu, act_hidden=tf.nn.relu, pwlst=pwlst_item,
                                     en_BN=self.enable_bn, en_res=True)

            Hlst[-1]=tf.nn.relu(Hlst[-1])

            flat_H = tf.reshape(Hlst[-1], (-1, Hlst[-1].shape[1] * Hlst[-1].shape[2]))
            print('Input shape is:', X.shape)
            print('Last output shape is:', Hlst[-1].shape)
            print('Flattened last output shape is:', flat_H.shape)

            if len(self.dense_hid) > 1:
                dense_layers = [tf.layers.dense(inputs=flat_H, units=self.dense_hid[0], activation=tf.nn.relu,
                                                kernel_initializer='glorot_uniform', use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

                for n in range(1, len(self.dense_hid[:-1])):
                    dense_layers.append(tf.layers.dense(inputs=dense_layers[n - 1], units=self.dense_hid[n], activation=tf.nn.relu,
                                                     kernel_initializer='glorot_uniform', use_bias=True,
                                                     bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

                dense_layers.append(tf.layers.dense(inputs=dense_layers[-1], units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer))

            else:
                dense_layers= [tf.layers.dense(inputs=flat_H, units=self.dense_hid[-1], activation=None,
                                                    kernel_initializer='glorot_uniform',
                                                    use_bias=True, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)]

            y_pred = dense_layers[-1]


            loss_dict={
                'MSE': tf.losses.mean_squared_error(y, y_pred),
                'MAE': tf.losses.absolute_difference(y, y_pred),
                'HUBER': tf.losses.huber_loss(y, y_pred, delta=1.0),
                'SCCLMAE': self.scclmae(y, y_pred, yid),
            }

            if self.is_classify:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred)
            else:
                loss = loss_dict[self.loss_type]

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss=tf.add_n([loss]+reg_losses)

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            train_op = opt.minimize(loss)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        model_param_dict={
            'X': X,
            'AdjM': AdjM,
            'y': y,
            'init': init,
            'train_op':train_op,
            'loss':loss,
            'y_pred': y_pred,
            'adjm_power':pwlst_hid,
            'alst_hid':alst_hid,
            'Wlst_hid':Wlst_hid,
            'blst_hid': blst_hid,
            'Hlst':Hlst,
            'dense_layers': dense_layers,
            'saver': saver,
            'X_in_var':data_param_dict_in['X_in_var'],
            'Xadjm_in_var': data_param_dict_in['Xadjm_in_var'],
            'Y_in_var': data_param_dict_in['Y_in_var'],
            'YID_in_var': data_param_dict_in['YID_in_var'],
            'val_X_in_var': data_param_dict_in['val_X_in_var'],
            'val_Xadjm_in_var': data_param_dict_in['val_Xadjm_in_var'],
            'val_Y_in_var': data_param_dict_in['val_Y_in_var'],
            'val_YID_in_var': data_param_dict_in['val_YID_in_var'],
            'shuffle_buff_size':data_param_dict_in['shuffle_buff_size'],
            'trn_iterator': data_param_dict_in['trn_iterator'],
            'val_iterator': data_param_dict_in['val_iterator']
        }

        return model_param_dict

    def train_model(self, trn_X, trn_AdjM, trn_y, trn_yid, val_X=None, val_AdjM=None, val_y=None, val_yid=None, restore_path=None, sbuff_size=100000):
        '''
        Train the model.

        Inputs:
        trn_X: Train dataset for X input features
        trn_AdjM: Train dataset for AdjM
        trn_y: Train dataset for y
        trn_yid: Train dataset for y labels
        val_X: Validation dataset for X input features
        val_AdjM: Validation dataset for AdjM
        val_y: Validation _dataset for y
        val_yid: Validation dataset for y labels
        restore_path: file path to a pre-trained model to initialize model
        sbuff_size: shuffle size for dataset

        Returns:
        A dict containing trained model variables.

        '''
        reset_graph()

        self.istrain=True
        self.batch_count = int(np.ceil(len(trn_X) / self.batch_size))

        print('Train dataset batch count is', self.batch_count)
        print('Train dataset shuffle size is', sbuff_size)

        with tf.Session(graph=self.gat_graph) as sess:

            if restore_path is None:
                self.init.run()
            else:
                self.saver.restore(sess, restore_path)

            train_writer = tf.summary.FileWriter(self.sum_folder, sess.graph, flush_secs=120)

            loss_lst=[]
            loss_val_lst=[]
            y_val_best, val_pwlst_best, val_alst_best, val_Wlst_best, val_blst_best, val_Hlst_best, val_dense_layers_best = [None] * 7

            for i in range(self.num_epochs):
                print("Running epoch:", i)

                sess.run(self.trn_iterator, feed_dict={self.X_in_var: trn_X, self.Xadjm_in_var: trn_AdjM,
                                        self.Y_in_var: trn_y, self.YID_in_var:trn_yid, self.sbuff_size: sbuff_size})
                loss_batch_tot=0
                for j in range(self.batch_count):
                    _, loss_batch = sess.run([self.train_op, self.loss])
                    loss_batch_tot+=loss_batch

                loss_ep = loss_batch_tot/self.batch_count
                print("Loss per epoch", i, ":", loss_ep)
                loss_lst.append(loss_ep)

                if val_X is not None:
                    if not((i+1) % self.early_stop_int) and ((i+1) > self.early_stop_threshold):
                        model_trn_in_keep_prob = self.trn_in_keep_prob
                        model_trn_eij_keep_prob = self.trn_eij_keep_prob
                        self.trn_in_keep_prob = 1.0
                        self.trn_eij_keep_prob = 1.0
                        self.istrain = False

                        sess.run(self.val_iterator, feed_dict={self.val_X_in_var: val_X, self.val_Xadjm_in_var: val_AdjM,
                                                self.val_Y_in_var: val_y, self.val_YID_in_var: val_yid})
                        y_val = []
                        loss_val_tot=0
                        val_batch_count=0

                        val_sess_run_lst = [self.y_pred, self.loss, self.pwlst_hid, self.alst_hid, self.Wlst_hid, self.blst_hid,
                                            self.Hlst, self.dense_layers]

                        try:
                            while True:
                                y_valb, loss_valb, val_pwlst, val_alst, val_Wlst, val_blst, val_Hlst, val_dense_layers = sess.run(val_sess_run_lst)
                                y_val=y_val+list(y_valb)
                                loss_val_tot+=loss_valb
                                val_batch_count+=1
                        except tf.errors.OutOfRangeError:
                            print("Validation data processsed in", val_batch_count, "batches.")

                        loss_val=loss_val_tot/val_batch_count
                        loss_val_lst.append(loss_val)

                        if loss_val < self.loss_early_stop:
                            early_save_path = self.saver.save(sess, os.path.join(self.models_folder, "early_stop_model_params_int"
                                                                                 +str(self.early_stop_int)+"_th"+str(self.early_stop_threshold)+"_"+self.postfix+".chkpt"))
                            print("Validation loss is", loss_val, ", Train loss is", loss_ep, ", early stop threshold is",
                                  self.loss_early_stop, ". Saving model params in", early_save_path, "and continuing training")
                            self.loss_early_stop = loss_val

                            y_val_best=y_val
                            val_pwlst_best = val_pwlst
                            val_alst_best = val_alst
                            val_Wlst_best = val_Wlst
                            val_blst_best = val_blst
                            val_Hlst_best = val_Hlst
                            val_dense_layers_best = val_dense_layers

                        self.trn_in_keep_prob = model_trn_in_keep_prob
                        self.trn_eij_keep_prob = model_trn_eij_keep_prob
                        self.istrain = True

            save_path = self.saver.save(sess, "tmp_saved_models/final_epoch_model_params_ep" + str(i) + "_batch_" + str(j) + "_" +self.postfix+".chkpt")
            print("Final train loss is", loss_ep, ", early stop threshold was",
                  self.loss_early_stop, ". Saving model params in", save_path, "and exiting.")

        trained_model_vars={'train_losses': loss_lst,
                            'val_losses': loss_val_lst,
                            'val_y_pred': y_val_best,
                            'val_y_true':val_y,
                            'val_adjm power': val_pwlst_best,
                            'val_alst_hid':val_alst_best,
                            'val_Wlst_hid':val_Wlst_best,
                            'val_blst_hid':val_blst_best,
                            'val_Hlst':val_Hlst_best,
                            'val_dense_layers': val_dense_layers_best,
                            'early_stop_int': self.early_stop_int,
                            'postfix_label': self.postfix,
                            'shuffle_buff_size':sbuff_size
                            }
        cm.pklsave('saved_model_params/train_val_params_'+self.postfix+'.pkl', trained_model_vars)

        return trained_model_vars

    def model_predict(self, Xfm_in, Xadjm_in, YID_in, saved_model_chkpt_path=None, outfn='saved_model_params/'):
        '''
        Predicts from a trained model.

        Xfm_in: A dataset for X input features
        Xadjm_in: A dataset for adjacency matrix
        YID_in: A dataset for y label ids to be predicted
        saved_model_chkpt_path: file path to trained model parameters
        outfn: save path for predicted values and model data

        Returns:
        A dictionary with model variable and predicted values.
        '''

        reset_graph()
        if saved_model_chkpt_path is None:
            print('Please provide a trained model to restore for prediction.')
            return None


        with tf.Session(graph=self.gat_graph) as sess:

            model_trn_in_keep_prob = self.trn_in_keep_prob
            model_trn_eij_keep_prob = self.trn_eij_keep_prob
            self.trn_in_keep_prob = 1.0
            self.trn_eij_keep_prob = 1.0
            self.istrain = False

            self.saver.restore(sess, saved_model_chkpt_path)
            print('Restored model at:', saved_model_chkpt_path)

            dummy_Y_in=[np.zeros(shape=[self.cls_num])]*len(Xfm_in)

            sess.run(self.val_iterator,
                     feed_dict={self.val_X_in_var: Xfm_in, self.val_Xadjm_in_var: Xadjm_in,
                                self.val_Y_in_var: dummy_Y_in, self.val_YID_in_var: YID_in})

            y_tst = []
            tst_batch_count = 0

            tst_sess_run_lst = [self.y_pred, self.pwlst_hid, self.alst_hid, self.Wlst_hid, self.blst_hid,
                                self.Hlst, self.dense_layers]

            try:
                while True:
                    y_tstb, tst_pwlst, tst_alst, tst_Wlst, tst_blst, tst_Hlst, tst_dense_layers = sess.run(tst_sess_run_lst)
                    y_tst = y_tst + list(y_tstb)
                    tst_batch_count += 1
            except tf.errors.OutOfRangeError:
                print("Test data predicted in", tst_batch_count, "batches.")

            self.trn_in_keep_prob = model_trn_in_keep_prob
            self.trn_eij_keep_prob = model_trn_eij_keep_prob
            self.istrain = True

        infer_model_vars={'y_predicted': y_tst,
                    'tst_pwlst': tst_pwlst,
                    'tst_alst': tst_alst,
                    'tst_Wlst': tst_Wlst,
                    'tst_blst': tst_blst,
                    'tst_Hlst': tst_Hlst,
                    'tst_dense_layers': tst_dense_layers,
                    'postfix_label': self.postfix
        }
        cm.pklsave(outfn+'test_params_' + self.postfix + '.pkl', infer_model_vars)

        return infer_model_vars

    def model_validate(self, Xfm_in, Xadjm_in, Y_in, YID_in,  saved_model_chkpt_path=None, outfn='saved_model_params/'):
        '''
        Validates from a trained model.

        Xfm_in: A dataset for X input features
        Xadjm_in: A dataset for adjacency matrix
        Y_in: A dataset for y values
        YID_in: A dataset for y label ids to be predicted
        saved_model_chkpt_path: file path to trained model parameters
        outfn: save path for predicted values and model data

        Returns:
        A dictionary with model variables and predicted values.
        '''

        reset_graph()
        if saved_model_chkpt_path is None:
            print('Please provide a trained model to restore for validation.')
            return None


        with tf.Session(graph=self.gat_graph) as sess:
            model_trn_in_keep_prob = self.trn_in_keep_prob
            model_trn_eij_keep_prob = self.trn_eij_keep_prob
            self.trn_in_keep_prob = 1.0
            self.trn_eij_keep_prob = 1.0
            self.istrain = False

            self.saver.restore(sess, saved_model_chkpt_path)
            print('Restored model at:', saved_model_chkpt_path)

            sess.run(self.val_iterator, feed_dict={self.val_X_in_var: Xfm_in, self.val_Xadjm_in_var: Xadjm_in,
                                                   self.val_Y_in_var: Y_in, self.val_YID_in_var: YID_in})
            y_val = []
            loss_val_tot = 0
            val_batch_count = 0
            val_sess_run_lst = [self.y_pred, self.loss, self.pwlst_hid, self.alst_hid, self.Wlst_hid, self.blst_hid,
                                self.Hlst, self.dense_layers]

            try:
                while True:
                    y_valb, loss_valb, val_pwlst, val_alst, val_Wlst, val_blst, val_Hlst, val_dense_layers = sess.run(
                        val_sess_run_lst)
                    y_val = y_val + list(y_valb)
                    loss_val_tot += loss_valb
                    val_batch_count += 1
            except tf.errors.OutOfRangeError:
                print("Validation data processsed in", val_batch_count, "batches.")

            loss_val = loss_val_tot / val_batch_count

            self.trn_in_keep_prob = model_trn_in_keep_prob
            self.trn_eij_keep_prob = model_trn_eij_keep_prob
            self.istrain = True

        validate_model_vars = {'y_predicted': y_val,
                               'y_true':Y_in,
                               'loss_val':loss_val,
                            'val_pwlst': val_pwlst,
                            'val_alst': val_alst,
                            'val_Wlst': val_Wlst,
                            'val_blst': val_blst,
                            'val_Hlst': val_Hlst,
                            'val_dense_layers': val_dense_layers,
                            'postfix_label': self.postfix
                            }
        cm.pklsave(outfn+'validate_params_' + self.postfix + '.pkl', validate_model_vars)

        return validate_model_vars


def reset_graph(seed=1234):
    '''
    resets default tensorflow graph
    sets random seed for numpy
    '''
    tf.reset_default_graph()
    np.random.seed(seed)

def set_graph_seed(seed=1234):
    '''
    sets random seed at graph level.
    '''
    tf.set_random_seed(seed)
