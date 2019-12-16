'''
Common methods for the project
'''

import pickle



def pklsave(filename,data):
    '''
    A simple wrapper for pickling data
    Inputs:
    filename: file name for pickle data
    data: python object to be saved

    Returns:
    Saves data in filename
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('data saved in', filename, 'as pickle file in current folder')

def pklload(filename):
    '''
    A simple wrapper to load object from pickle file
    Inputs:
    filename: file name for pickled data
    
    Returns:
    A python object loaded from filename
    '''
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    print(filename, 'data loaded into python object')
    return data
