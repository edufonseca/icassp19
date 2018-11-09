
import numpy as np
import os, re
import matplotlib
matplotlib.use('agg')


#########################################################################
# Some of these functions have been inspired on a framework by Marius Miron developed for a pydata workshop
# https://github.com/nkundiushuti/pydata2017bcn/blob/master/util.py
#########################################################################


def save_tensor(var, out_path=None, suffix='_mel'):
    """
    Saves a numpy array as a binary file
    -review the shape saving when it is a label
    """
    assert os.path.isdir(os.path.dirname(out_path)), "path to save tensor does not exist"
    var.tofile(out_path.replace('.data', suffix + '.data'))
    save_shape(out_path.replace('.data', suffix + '.shape'), var.shape)


def load_tensor(in_path, suffix=''):
    """
    Loads a binary .data file
    """
    assert os.path.isdir(os.path.dirname(in_path)), "path to load tensor does not exist"
    f_in = np.fromfile(in_path.replace('.data', suffix + '.data'))
    shape = get_shape(in_path.replace('.data', suffix + '.shape'))
    f_in = f_in.reshape(shape)
    return f_in


def save_shape(shape_file, shape):
    """
    Saves the shape of a numpy array
    """
    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')


def get_shape(shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file')


def get_num_instances_per_file(f_name, patch_len=25, patch_hop=12):
    """
    Return the number of context_windows or instances generated out of a given file
    """
    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    file_frames = float(shape[0])
    return np.maximum(1, int(np.ceil((file_frames-patch_len)/patch_hop)))


def get_feature_size_per_file(f_name):
    """
    Return the dimensionality of the features in a given file.
    Typically, this will be the number of bins in a T-F representation
    """
    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    return shape[1]


def make_sure_isdir(pre_path, _out_file):
    """
    make sure the a directory at the end of pre_path exists. Else create it
    :param pre_path:
    :param args:
    :return:
    """
    full_path = os.path.join(pre_path, _out_file)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path
