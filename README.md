## CoulGAT: A Graph Attention Framework with screened Coulomb Attention Mechanism

This repository is the implementation of graph attention framework and attention mechanism detailed in [CoulGAT: An Experiment on Interpretability of Graph Attention Networks.](https://arxiv.org/abs/1912.08409)

**Key Features:**

- Scalable and flexible model construction for deep plain and resnet architectures.
- Model_1: Plain CoulGAT with pooling option for final layer.
- Model_2: Resnet CoulGAT with pooling option for final layer.
- Model_3: Plain CoulGAT composed of attention layer blocks with pooling at end of each block.
- Model_4: Resnet CoulGAT with pooling at the end of each resnet block.
- SCCLMAE loss for nonzero labels only or MSE/Huber/MAE loss for all labels
- Uses Adam optimizer
- Supports Early Stopping
- Dropout regularization for attention layers
- L2 regularization for dense layers
- Flexible hyperparameter optimization through a parameter dictionary
- Data preprocessing classes for [Predicting Molecular Properties dataset from Kaggle.](https://www.kaggle.com/c/champs-scalar-coupling/data)
- Support for keeping track of model parameters and learned attention layers in memory and as pickle file.

**Parameter Dictionary Examples to construct a CoulGAT model:**

A 3-layer Plain CoulGAT model with pooling as last layer:

```python
my_param_dict_model_1={
   'num_nodes':29, #N: max count of nodes in input X
    'num_features':211, #F: number of input features in X
    'class_number':29*211, #length of output vector
    'num_graph_layers': 4, #Attention Layer Count+1 (for input)
    'list_hidden_graph_layers':[211, 422, 422, 422], #first element is input F
    'list_hidden_heads':[1,5,5,5], #first element is 1 for input
    'batch_size': 128,
    'num_epochs':200,
    'learning_rate':0.001,
    'reg_scale':0.0005, #L2 regularization
    'loss_type':'SCCLMAE', #'MSE' for mse loss
    'trn_in_keep_prob': 1.0,  #dropout keep probability 1
    'trn_eij_keep_prob': 1.0, #dropout keep probability 2
    'enable_pw': True, #enable learnable power matrix
    'is_classify': False,
    'early_stop_threshold':None, #start early stopping when epoch count is larger than this value
    'num_early_stop':20, #number of validation points to collect
    'models_folder':'tmp_saved_models', 
    'sum_folder': 'summaries', #for tensorboard events
    'label': 'my_CoulGAT_plain_model'
    'resgnn_block_num': 2, #number of attention layers in one res-block
    'use_head_averaging': True, #enable pooling as last layer
    'enable_bn': False, #enable batch normalization for resnet models
    'bn_momentum': 0.99, 
    'model_name': 'model_1' #name of model to call
}
```
A 140 layer resnet CoulGAT model without pooling as final layer:

```python
my_param_dict_model_2={
   'num_nodes':29,
    'num_features':211,
    'class_number':29*211,
    'num_graph_layers': 71, #number of resnet blks + 1 (for input)
    'list_hidden_graph_layers':50, #one hidden feature number for all attention layers 
    'list_hidden_heads':5, #one head count for all attention layers
    'batch_size': 128,
    'num_epochs':200, 
    'learning_rate':0.001,
    'reg_scale':0.0005,
    'loss_type':'SCCLMAE',
    'trn_in_keep_prob': 1.0, 
    'trn_eij_keep_prob': 1.0,
    'enable_pw': True,
    'is_classify': False,
    'early_stop_threshold':None,
    'num_early_stop':20,
    'models_folder':'tmp_saved_models',
    'sum_folder': 'summaries',
    'label': 'my_CoulGAT_resnet_model',
    'resgnn_block_num': 2,
    'use_head_averaging': False,
    'enable_bn': True,
    'bn_momentum':0.99,
    'model_name': 'model_2'
}
```

A 16-layer resnet CoulGAT model with final pooling at each resnet block:

```python
my_param_dict_model_4={
   'num_nodes':29,
    'num_features':211,
    'class_number':29*211,
    'num_graph_layers': 9, #number of attention layer blocks + 1 
    'list_hidden_graph_layers':[211,211], #the feature sizes for single block
    'list_hidden_heads':[5,5], #head counts for single block
    'batch_size': 128,
    'num_epochs':200,
    'learning_rate':0.001,
    'reg_scale':0.0005,
    'loss_type':'SCCLMAE',
    'trn_in_keep_prob': 1.0,
    'trn_eij_keep_prob': 1.0,
    'enable_pw': True,
    'is_classify': False,
    'early_stop_threshold':None,
    'num_early_stop':20,
    'models_folder':'tmp_saved_models',
    'sum_folder': 'summaries',
    'label': 'my_CoulGAT_model_4',
    'resgnn_block_num': 2,
    'use_head_averaging': True, #model_3/model_4 ignores this value
    'enable_bn': True,
    'bn_momentum':0.99,
    'model_name': 'model_4'
}
```

Sample run:

```python
import gatt_model as gm
my_model=gm.GattModel(my_param_dict_model_1)
train_out=my_model.train_model(...train/val data...)
```

**Local folders that need to be created:**

- "saved_model_params" : for saving hyperparameters and train and validation results.
- "tmp_prep_data": for saving temporary files in data preparation.
- Other folders defined in parameter dictionary or passed as input.