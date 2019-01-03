
import numpy as np
import os
import utils
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence, to_categorical

# NOTE:
# these data generators work for small-medium size datasets under no memory constraints, eg RAM 32GB or more.
# If used with smaller RAMs, a slightly different approach for feeding the net may be needed.


def get_label_files(filelist=None, dire=None, suffix_in=None, suffix_out=None):
    """

    :param filelist:
    :param dire:
    :param suffix_in:
    :param suffix_out:
    :return:
    """

    nb_files_total = len(filelist)
    labels = np.zeros((nb_files_total, 1), dtype=np.float32)
    for f_id in range(nb_files_total):
        labels[f_id] = utils.load_tensor(in_path=os.path.join(dire, filelist[f_id].replace(suffix_in, suffix_out)))
    return labels


class DataGeneratorPatch(Sequence):
    """
    Reads data from disk and returns batches.
    """

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.list_fnames = file_list
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        # - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)

            # standardize the data
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if train set, create scaler, fit, transform, and save the scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)
                # this scaler will be used later on to scale val and test data

            else:
                # if we are in val or test set, load the training scaler as a param and transform
                self.features2d = scaler.transform(self.features2d)

            # after scaling in 2D, go back to tensor
            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        # but all the patches are contiguously ordered. shuffle them before making batches
        self.on_epoch_end()
        self.n_classes = params_learn.get('n_classes')

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        - shuffle them
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')
        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances (or T_F patches) per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        # how many batches can we fit in the set
        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features and labels
        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):
        """
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features
        :param f_id:
        :return:
        """

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

    def __len__(self):
        return self.nb_iterations

    def __getitem__(self, index):
        """
        takes an index (batch number) and returns one batch of self.batch_size
        :param index:
        :return:
        """
        # index is taken care of by the Sequencer inherited
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # fetch labels for the batch
        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])
        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        # fetch features for the batch and adjust format to input CNN
        # (batch_size, 1, time, freq) for channels_first
        features = self.features[indexes, np.newaxis]
        return features, y_cat

    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.nb_inst_total)


class PatchGeneratorPerFile(object):
    """
    Reads whole T_F representations from disk,
    and stores T_F patches *for a given entire file* in a tensor
    typically for prediction on a test set

    """

    def __init__(self, feature_dir=None, file_list=None, params_extract=None,
                 suffix_in='_mel', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        if feature_dir is not None:
            self.get_patches_features(feature_dir, file_list)

            # standardize the data: assuming this is used for inference
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if we are in val or test subset, load the training scaler as a param and transform
            self.features2d = scaler.transform(self.features2d)

            # go back to 3D tensor
            self.features = self.features2d.reshape(self.nb_patch_total, self.patch_len, self.feature_size)

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows or instances generated out of a given file
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"

        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data')]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_patch_total = self.nb_inst_cum[-1]

        # init current file, to keep track of the file yielded for prediction
        self.current_f_idx = 0

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features
        self.features = np.zeros((self.nb_patch_total, self.patch_len, self.feature_size), dtype=self.floatx)

        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):
        """
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features
        :param f_id:
        :return:
        """

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]  # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]  # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

    def get_patches_file(self):
        """
        Returns all the patches for one single audio clip
        """

        self.current_f_idx += 1
        # ranges form 1 to self.nb_files (ignores 0)
        assert self.current_f_idx <= self.nb_files, 'All the test files have been dispatched'

        # fetch features in the batch and adjust format to input CNN
        # (nb_patches_per_file, 1, time, freq)
        features = self.features[self.nb_inst_cum[self.current_f_idx-1]: self.nb_inst_cum[self.current_f_idx], np.newaxis]
        return features


class DataGeneratorPatchOrigin(Sequence):

    """
    Reads data from disk and returns batches.
    allows to create one-hot encoded vectors carrying flags, ie 100 instead of 1.
    this is used in the loss functions to distinguish patches coming from noisy or clean set

    """

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.list_fnames = file_list
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))
        self.noisy_ids = params_learn.get('noisy_ids')

        # Given a directory with precomputed features in files:
        # - create the variable self.features with all the TF patches of all the files in the feature_dir
        # - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)

            # standardize the data
            self.features2d = self.features.reshape(-1, self.features.shape[2])

            # if train set, create scaler, fit, transform, and save the scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)
                # this scaler will be used later on to scale val and test data

            else:
                # if we are in val or test set, load the training scaler as a param and transform
                self.features2d = scaler.transform(self.features2d)

            # after scaling in 2D, go back to tensor
            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        self.on_epoch_end()
        self.n_classes = params_learn.get('n_classes')

    def get_num_instances_per_file(self, f_name):
        """
        Return the number of context_windows, patches, or instances generated out of a given file
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):
        """
        Return the dimensionality of the features in a given file.
        Typically, this will be the number of bins in a T-F representation
        """
        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):
        """
        Given a directory with precomputed features in files:
        - create the variable self.features with all the TF patches of all the files in the feature_dir
        - create the variable self.labels with the corresponding labels (at patch level, inherited from file)
        - shuffle them
        """
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')
        # list of file names containing features
        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        # For all set, cumulative sum of instances (or T_F patches) per file
        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        # how many batches can we fit in the set
        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        # feature size (last dimension of the output)
        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        # init the variables with features and labels
        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)
        # analogous column vector to flag patches coming from noisy subset of train data
        # init to 0. Only 1 if they come from noisy subset
        self.noisy_patches = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)


        # fetch all data from hard-disk
        for f_id in range(self.nb_files):
            # for every file in disk, perform slicing into T-F patches, and store them in tensor self.features
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):
        """
        # for a file specified by id,
        # perform slicing into T-F patches, and store them in tensor self.features

        :param f_id:
        :return:
        """

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        # indexes to store patches in self.features, according to the nb of instances from the file
        idx_start = self.nb_inst_cum[f_id]      # start for a given file
        idx_end = self.nb_inst_cum[f_id + 1]    # end for a given file

        # slicing + storing in self.features
        # copy each TF patch of size (context_window_frames,feature_size) in self.features
        idx = 0  # to index the different patches of f_id within self.features
        start = 0  # starting frame within f_id for each T-F patch
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            # update indexes
            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

        if int(self.file_list[f_id].split('_')[0]) in self.noisy_ids:
            # if the clip comes from noisy subset, flag to 1 all its patches
            self.noisy_patches[idx_start: idx_end] = 1

    def __len__(self):
        return self.nb_iterations

    def __getitem__(self, index):
        """
        takes an index (batch number) and returns one batch of self.batch_size

        :param index:
        :return:
        """
        # index is taken care of by the Sequencer inherited
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # fetch labels for the batch
        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])

        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        # tune the one-hot vectors of the patches coming from clips in the noisy subset
        for tt in np.arange(self.batch_size):
            if self.noisy_patches[indexes[tt]] == 1:
                y_cat[tt] *= 100

        # fetch features for the batch and adjust format to input CNN
        # (batch_size, 1, time, freq) for channels_first
        features = self.features[indexes, np.newaxis]
        return features, y_cat

    def on_epoch_end(self):
        # shuffle data between epochs
        self.indexes = np.random.permutation(self.nb_inst_total)
