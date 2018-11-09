

#########################################################################
# Copyright Eduardo Fonseca 2018, v1.0
# This software is distributed under the terms of the License GNU AFFERO GENERAL PUBLIC LICENSE
#
# If you use this code or part of it, please cite the following paper:
# Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra, "Learning Sound Event
# Classifiers from Web Audio with Noisy Labels", in Proc. IEEE ICASSP 2019, Brighton, UK, 2019
#
#########################################################################

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import utils
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from data import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile, DataGeneratorPatchOrigin
from architectures import get_model_baseline
from eval import Evaluator
from losses import lq_loss_wrap, crossentropy_max_wrap, crossentropy_outlier_wrap, crossentropy_reed_wrap,\
    crossentropy_max_origin_wrap, crossentropy_outlier_origin_wrap, lq_loss_origin_wrap, crossentropy_reed_origin_wrap


start = time.time()

now = datetime.datetime.now()
print("Current date and time:")
print(str(now))

# =========================================================================================================

# ==================================================================== ARGUMENTS
parser = argparse.ArgumentParser(description='Code for ICASSP2019 paper Learning Sound Event Classifiers from Web Audio'
                                             ' with Noisy Labels')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))


# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables

# Read parameters file from yaml passed by argument
params = yaml.load(open(args.params_yaml))
params_ctrl = params['ctrl']
params_extract = params['extract']
params_learn = params['learn']
params_loss = params['loss']
params_recog = params['recognizer']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')


# determine loss function
flag_origin = False
if params_loss.get('type') == 'CCE':
    params_loss['type'] = 'categorical_crossentropy'
elif params_loss.get('type') == 'lq_loss':
    params_loss['type'] = lq_loss_wrap(params_loss.get('q_loss'))
elif params_loss.get('type') == 'CCE_max':
    params_loss['type'] = crossentropy_max_wrap(params_loss.get('m_loss'))
elif params_loss.get('type') == 'CCE_outlier':
    params_loss['type'] = crossentropy_outlier_wrap(params_loss.get('l_loss'))
elif params_loss.get('type') == 'bootstrapping':
    params_loss['type'] = crossentropy_reed_wrap(params_loss.get('reed_beta'))

# selective loss based on data origin
elif params_loss.get('type') == 'CCE_max_origin':
    params_loss['type'] = crossentropy_max_origin_wrap(params_loss.get('m_loss'))
    flag_origin = True
elif params_loss.get('type') == 'CCE_outlier_origin':
    params_loss['type'] = crossentropy_outlier_origin_wrap(params_loss.get('l_loss'))
    flag_origin = True
elif params_loss.get('type') == 'lq_loss_origin':
    params_loss['type'] = lq_loss_origin_wrap(params_loss.get('q_loss'))
    flag_origin = True
elif params_loss.get('type') == 'bootstrapping_origin':
    params_loss['type'] = crossentropy_reed_origin_wrap(params_loss.get('reed_beta'))
    flag_origin = True


params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))
#

# ======================================================== PATHS FOR DATA, FEATURES and GROUND TRUTH
# where to look for the dataset
path_root_data = os.path.join('/data', params_ctrl.get('dataset'))

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               'featuredir_tr': 'audio_train_varup2/',
               'featuredir_te': 'audio_test_varup2/',
               'path_to_dataset': path_root_data,
               'audiodir_tr': 'audio_train/',
               'audiodir_te': 'audio_test/',
               'audio_shapedir_tr': 'audio_train_shapes/',
               'audio_shapedir_te': 'audio_test_shapes/',
               'gt_files': os.path.join('ground_truth_csvs', params_ctrl.get('dataset'))}


params_path['featurepath_tr'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_tr'))
params_path['featurepath_te'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_te'))

params_path['audiopath_tr'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_tr'))
params_path['audiopath_te'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_te'))

params_path['audio_shapepath_tr'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_tr'))
params_path['audio_shapepath_te'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_te'))


# ======================================================== SPECIFIC PATHS TO SOME IMPORTANT FILES
# ground truth, load model, save model, predictions, results
params_files = {'gt_test': os.path.join(params_path.get('gt_files'), 'test.csv'),
                'gt_train': os.path.join(params_path.get('gt_files'), 'train.csv')}

# # ============================================= print all params to keep record in output file
print('\nparams_ctrl=')
pprint.pprint(params_ctrl, width=1, indent=4)
print('params_files=')
pprint.pprint(params_files, width=1, indent=4)
print('params_extract=')
pprint.pprint(params_extract, width=1, indent=4)
print('params_learn=')
pprint.pprint(params_learn, width=1, indent=4)
print('params_loss=')
pprint.pprint(params_loss, width=1, indent=4)
print('params_recog=')
pprint.pprint(params_recog, width=1, indent=4)
print('\n')


# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA

# aim: lists with all wav files for tr and te
train_csv = pd.read_csv(params_files.get('gt_train'))
test_csv = pd.read_csv(params_files.get('gt_test'))
filelist_audio_tr = train_csv.fname.values.tolist()
filelist_audio_te = test_csv.fname.values.tolist()

# get positions of manually_verified clips: separate between CLEAN and NOISY sets
filelist_audio_tr_flagveri = train_csv.manually_verified.values.tolist()
idx_flagveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 1]
idx_flagnonveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 0]

# create list of ids that come from the noisy set
noisy_ids = [int(filelist_audio_tr[i].split('.')[0]) for i in idx_flagnonveri]
params_learn['noisy_ids'] = noisy_ids

# get positions of noisy_small_duration and noisy_small_clips:
# subsets of the NOISY set of comparable size to that of CLEAN
filelist_audio_tr_nV_small_dur = train_csv.nV_small_dur.values.tolist()
idx_nV_small_dur = [i for i, x in enumerate(filelist_audio_tr_nV_small_dur) if x == 1]

filelist_audio_tr_nV_small_clips = train_csv.nV_small_clips.values.tolist()
idx_nV_small_clips= [i for i, x in enumerate(filelist_audio_tr_nV_small_clips) if x == 1]


# create dict with ground truth mapping with labels:
# -key: path to wav
# -value: the ground truth label too
file_to_label = {params_path.get('audiopath_tr') + k: v for k, v in
                 zip(train_csv.fname.values, train_csv.label.values)}

# ========================================================== CREATE VARS FOR DATASET MANAGEMENT
# list with unique n_classes labels and aso_ids
list_labels = sorted(list(set(train_csv.label.values)))
list_aso_ids = sorted(list(set(train_csv.aso_id.values)))

# create dicts such that key: value is as follows
# label: int
# int: label
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

# create ground truth mapping with categorical values
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

#
#
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# ========================================================== FEATURE EXTRACTION
# compute T_F representation
# mel-spectrogram for all files in the dataset and store it

if params_ctrl.get('feat_ext'):
    n_extracted_tr = 0; n_extracted_te = 0; n_failed_tr = 0; n_failed_te = 0

    # only if features have not been extracted, ie
    # if folder does not exist, or it exists with less than 80% of the feature files
    # create folder and extract features
    nb_files_tr = len(filelist_audio_tr)
    if not os.path.exists(params_path.get('featurepath_tr')) or \
                    len(os.listdir(params_path.get('featurepath_tr'))) < nb_files_tr*0.8:
        os.makedirs(params_path.get('featurepath_tr'))
        os.makedirs(params_path.get('featurepath_te'))

        for idx, f_name in enumerate(filelist_audio_tr):
            f_path = os.path.join(params_path.get('audiopath_tr'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_mel')

                # save also label
                utils.save_tensor(var=np.array([file_to_int[f_path]], dtype=float),
                                  out_path=os.path.join(params_path.get('featurepath_tr'),
                                                        f_name.replace('.wav', '.data')), suffix='_label')

                if os.path.isfile(os.path.join(params_path.get('featurepath_tr'),
                                               f_name.replace('.wav', suffix_in + '.data'))):
                    n_extracted_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted tr features', (idx + 1), nb_files_tr, f_path))
                else:
                    n_failed_tr += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract tr features', (idx + 1), nb_files_tr, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this tr audio is in the csv but not in the folder', (idx + 1), nb_files_tr, f_path))

        print('n_extracted_tr: {0} / {1}'.format(n_extracted_tr, nb_files_tr))
        print('n_failed_tr: {0} / {1}\n'.format(n_failed_tr, nb_files_tr))

        nb_files_te = len(filelist_audio_te)
        for idx, f_name in enumerate(filelist_audio_te):
            f_path = os.path.join(params_path.get('audiopath_te'), f_name)
            if os.path.isfile(f_path) and f_name.endswith('.wav'):
                # load entire audio file and modify variable length, if needed
                y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
                y = modify_file_variable_length(data=y,
                                                input_fixed_length=params_extract['audio_len_samples'],
                                                params_extract=params_extract)

                # compute log-scaled mel spec. row x col = time x freq
                # this is done only for the length specified by loading mode (fix, varup, varfull)
                mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

                # save the T_F rep to a binary file (only the considered length)
                utils.save_tensor(var=mel_spectrogram,
                                  out_path=os.path.join(params_path.get('featurepath_te'),
                                                          f_name.replace('.wav', '.data')), suffix='_mel')

                if os.path.isfile(os.path.join(params_path.get('featurepath_te'),
                                               f_name.replace('.wav', '_mel.data'))):
                    n_extracted_te += 1
                    print('%-22s: [%d/%d] of %s' % ('Extracted te features', (idx + 1), nb_files_te, f_path))
                else:
                    n_failed_te += 1
                    print('%-22s: [%d/%d] of %s' % ('FAILING to extract te features', (idx + 1), nb_files_te, f_path))
            else:
                print('%-22s: [%d/%d] of %s' % ('this te audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))

        print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
        print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))


#
#
# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION
# ============================================================BATCH GENERATION

# Assuming features or T-F representations on a per-file fashion previously computed and in disk
# input: '_mel'
# output: '_label'

# select the subset of training data to consider: all, clean, noisy, noisy_small_dur
if params_ctrl.get('train_data') == 'all':
    ff_list_tr = [f for f in os.listdir(params_path.get('featurepath_tr')) if f.endswith(suffix_in + '.data') and
                  os.path.isfile(os.path.join(params_path.get('featurepath_tr'), f.replace(suffix_in, suffix_out)))]

elif params_ctrl.get('train_data') == 'clean':
    # only files (not path), feature file list for tr, only those that are manually verified: CLEAN SET
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_flagveri]

elif params_ctrl.get('train_data') == 'noisy':
    # only files (not path), feature file list for tr, only those that are NOT verified: NOISY SET
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_flagnonveri]

elif params_ctrl.get('train_data') == 'noisy_small_dur':
    # only files (not path), feature file list for tr, only a small portion of the NOISY SET
    # (comparable to CLEAN SET in terms of duration)
    ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in idx_nV_small_dur]

# get label for every file *from the .data saved in disk*, in float
labels_audio_train = get_label_files(filelist=ff_list_tr,
                                     dire=params_path.get('featurepath_tr'),
                                     suffix_in=suffix_in,
                                     suffix_out=suffix_out
                                     )

# sanity check
print('Number of clips considered as train set: {0}'.format(len(ff_list_tr)))
print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

# split the val set randomly (but stratified) within the train set
tr_files, val_files = train_test_split(ff_list_tr,
                                       test_size=params_learn.get('val_split'),
                                       stratify=labels_audio_train,
                                       random_state=42
                                       )

# to improve data generator
if flag_origin:
    tr_gen_patch = DataGeneratorPatchOrigin(feature_dir=params_path.get('featurepath_tr'),
                                      file_list=tr_files,
                                      params_learn=params_learn,
                                      params_extract=params_extract,
                                      suffix_in='_mel',
                                      suffix_out='_label',
                                      floatx=np.float32
                                      )
else:
    tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                      file_list=tr_files,
                                      params_learn=params_learn,
                                      params_extract=params_extract,
                                      suffix_in='_mel',
                                      suffix_out='_label',
                                      floatx=np.float32
                                      )

# to improve data generator
if flag_origin:
    val_gen_patch = DataGeneratorPatchOrigin(feature_dir=params_path.get('featurepath_tr'),
                                       file_list=val_files,
                                       params_learn=params_learn,
                                       params_extract=params_extract,
                                       suffix_in='_mel',
                                       suffix_out='_label',
                                       floatx=np.float32,
                                       scaler=tr_gen_patch.scaler
                                       )

else:
    val_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                       file_list=val_files,
                                       params_learn=params_learn,
                                       params_extract=params_extract,
                                       suffix_in='_mel',
                                       suffix_out='_label',
                                       floatx=np.float32,
                                       scaler=tr_gen_patch.scaler
                                       )


# ============================================================DEFINE AND FIT A MODEL
# ============================================================DEFINE AND FIT A MODEL
# ============================================================DEFINE AND FIT A MODEL

tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
# ============================================================
if params_ctrl.get('learn'):

    model = get_model_baseline(params_learn=params_learn, params_extract=params_extract)

    opt = Adam(lr=params_learn.get('lr'))
    model.compile(optimizer=opt, loss=params_loss.get('type'), metrics=['accuracy'])
    model.summary()

    # callbacks
    early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
    callback_list = [early_stop, reduce_lr]

    hist = model.fit_generator(tr_gen_patch,
                               steps_per_epoch=tr_gen_patch.nb_iterations,
                               epochs=params_learn.get('n_epochs'),
                               validation_data=val_gen_patch,
                               validation_steps=val_gen_patch.nb_iterations,
                               class_weight=None,
                               workers=4,
                               verbose=2,
                               callbacks=callback_list)


# ==================================================================================================== PREDICT
# ==================================================================================================== PREDICT
# ==================================================================================================== PREDICT

print('\nCompute predictions on test set:==================================================\n')

list_preds = []

te_files = [f for f in os.listdir(params_path.get('featurepath_te')) if f.endswith(suffix_in + '.data')]
# to store predictions
te_preds = np.empty((len(te_files), params_learn.get('n_classes')))

# grab every T_F rep file (computed on the file level)
# split it in T_F patches and store it in tensor, sorted by file
te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_te'),
                                     file_list=te_files,
                                     params_extract=params_extract,
                                     suffix_in='_mel',
                                     floatx=np.float32,
                                     scaler=tr_gen_patch.scaler
                                     )

for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
    # return all patches for a sound file
    patches_file = te_gen_patch.get_patches_file()

    # predicting now on the T_F patch level (not on the wav clip-level)
    preds_patch_list = model.predict(patches_file).tolist()
    preds_patch = np.array(preds_patch_list)

    # aggregate softmax values across patches in order to produce predictions on the file/clip level
    if params_learn.get('predict_agg') == 'amean':
        preds_file = np.mean(preds_patch, axis=0)
    elif params_recog.get('aggregate') == 'gmean':
        preds_file = gmean(preds_patch, axis=0)
    else:
        print('unkown aggregation method for prediction')
    te_preds[i, :] = preds_file


list_labels = np.array(list_labels)
pred_label_files_int = np.argmax(te_preds, axis=1)
pred_labels = [int_to_label[x] for x in pred_label_files_int]

# create dataframe with predictions
# columns: fname & label
# this is based on the features file, instead on the wav file (extraction errors could occur)
te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in os.listdir(params_path.get('featurepath_te'))
                if f.endswith(suffix_in + '.data')]
pred = pd.DataFrame(te_files_wav, columns=["fname"])
pred['label'] = pred_labels

#
# # =================================================================================================== EVAL
# # =================================================================================================== EVAL
# # =================================================================================================== EVAL
print('\nEvaluate ACC and print score============================================================================')

# read ground truth
gt_test = pd.read_csv(params_files.get('gt_test'))

# init Evaluator object
evaluator = Evaluator(gt_test, pred, list_labels, params_ctrl, params_files)

print('\n=============================ACCURACY===============================================================')
print('=============================ACCURACY===============================================================\n')
evaluator.evaluate_acc()
evaluator.evaluate_acc_classwise()
evaluator.print_summary_eval()

end = time.time()
print('\n=============================Job finalized==========================================================\n')
print('Time elapsed for the job: %7.2f hours' % ((end - start) / 3600.0))
print('\n====================================================================================================\n')
