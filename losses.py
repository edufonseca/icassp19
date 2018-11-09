
from keras import backend as K
import tensorflow as tf


def crossentropy_reed_wrap(_beta):
    def crossentropy_reed_core(y_true, y_pred):
        """
        This loss function is proposed in:
        Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping", 2014

        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_beta)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        # use predicted class proba directly to generate regression targets
        y_true_update = _beta * y_true + (1 - _beta) * y_pred

        # (2) compute loss as always
        _loss = -K.sum(y_true_update * K.log(y_pred), axis=-1)

        return _loss
    return crossentropy_reed_core


def lq_loss_wrap(_q):
    def lq_loss_core(y_true, y_pred):
        """
        This loss function is proposed in:
         Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
         Noisy Labels", 2018
        https://arxiv.org/pdf/1805.07836.pdf
        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_q)

        _tmp = y_pred * y_true
        _loss = K.max(_tmp, axis=-1)

        # compute the Lq loss between the one-hot encoded label and the prediction
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q

        return _loss
    return lq_loss_core


def crossentropy_max_wrap(_m):
    def crossentropy_max_core(y_true, y_pred):
        """
        This function is based on the one proposed in
        Il-Young Jeong and Hyungui Lim, "AUDIO TAGGING SYSTEM FOR DCASE 2018: FOCUSING ON LABEL NOISE,
         DATA AUGMENTATION AND ITS EFFICIENT LEARNING", Tech Report, DCASE 2018
        https://github.com/finejuly/dcase2018_task2_cochlearai

        :param y_true:
        :param y_pred:
        :return:
        """

        # hyper param
        print(_m)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # threshold
        t_m = K.max(_loss) * _m
        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32'))
        _loss = _loss * _mask_m

        return _loss
    return crossentropy_max_core


def crossentropy_outlier_wrap(_l):
    def crossentropy_outlier_core(y_true, y_pred):

        # hyper param
        print(_l)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median

            :param v:
            :return:
            """
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        # threshold
        t_l = _median_loss + _l*_std_loss
        _mask_l = 1 - (K.cast(K.greater(_loss, t_l), 'float32'))
        _loss = _loss * _mask_l

        return _loss
    return crossentropy_outlier_core



#########################################################################
# from here on we distinguish data points in the batch, based on its origin
# we only apply robustness measures to the data points coming from the noisy subset
# Therefore, the next functions are used only when training with the entire train set
#########################################################################


def crossentropy_reed_origin_wrap(_beta):
    def crossentropy_reed_origin_core(y_true, y_pred):
        # hyper param
        print(_beta)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # (1) dynamically update the targets based on the current state of the model: bootstrapped target tensor
        # use predicted class proba directly to generate regression targets
        y_true_bootstrapped = _beta * y_true + (1 - _beta) * y_pred

        # at this point we have 2 versions of y_true
        # decide which target label to use for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set
        _mask_noisy = K.reshape(_mask_noisy, (_y_true_shape[0], 1))
        _mask_clean = K.reshape(_mask_clean, (_y_true_shape[0], 1))

        # points coming from clean set use the standard true one-hot vector. dim is (batch_size, 1)
        # points coming from noisy set use the Reed bootstrapped target tensor
        y_true_final = y_true * _mask_clean + y_true_bootstrapped * _mask_noisy

        # (2) compute loss as always
        _loss = -K.sum(y_true_final * K.log(y_pred), axis=-1)

        return _loss
    return crossentropy_reed_origin_core


def lq_loss_origin_wrap(_q):
    def lq_loss_origin_core(y_true, y_pred):

        # hyper param
        print(_q)

        # 1) determine the origin of the patch, as a boolean vector in y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute two types of losses, for all the data points
        # (1) compute CCE loss for every data point
        _loss_CCE = -K.sum(y_true * K.log(y_pred), axis=-1)

        # (2) compute lq_loss for every data point
        _tmp = y_pred * y_true
        _loss_tmp = K.max(_tmp, axis=-1)
        # compute the Lq loss between the one-hot encoded label and the predictions
        _loss_q = (1 - (_loss_tmp + 10 ** (-8)) ** _q) / _q

        # decide which loss to take for each datapoint
        _mask_noisy = K.cast(_y_true_flag, 'float32')                   # only allows patches from noisy set
        _mask_clean = K.cast(K.equal(_y_true_flag, False), 'float32')   # only allows patches from clean set

        # points coming from clean set contribute with CCE loss
        # points coming from noisy set contribute with lq_loss
        _loss_final = _loss_CCE * _mask_clean + _loss_q * _mask_noisy

        return _loss_final
    return lq_loss_origin_core



def crossentropy_max_origin_wrap(_m):
    def crossentropy_max_origin_core(y_true, y_pred):

        # hyper param
        print(_m)

        # 1) determine the origin of the patch, as a boolean vector y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always TODO total or mask?
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        # threshold m
        t_m = K.max(_loss) * _m

        _mask_m = 1 - (K.cast(K.greater(_loss, t_m), 'float32') * K.cast(_y_true_flag, 'float32'))
        _loss = _loss * _mask_m

        return _loss
    return crossentropy_max_origin_core


def crossentropy_outlier_origin_wrap(_l):
    def crossentropy_outlier_origin_core(y_true, y_pred):

        # hyper param
        print(_l)

        # 1) determine the origin of the patch, as a boolean vector y_true_flag
        # (True = patch from noisy subset)
        _y_true_flag = K.greater(K.sum(y_true, axis=-1), 90)

        # 2) convert the input y_true (with flags inside) into a valid y_true one-hot-vector format
        # attenuating factor for data points that need it (those that came with a one-hot of 100)
        _mask_reduce = K.cast(_y_true_flag, 'float32') * 0.01

        # identity factor for standard one-hot vectors
        _mask_keep = K.cast(K.equal(_y_true_flag, False), 'float32')

        # combine 2 masks
        _mask = _mask_reduce + _mask_keep

        _y_true_shape = K.shape(y_true)
        _mask = K.reshape(_mask, (_y_true_shape[0], 1))

        # applying mask to have a valid y_true that we can use as always TODO total or mask?
        y_true = y_true * _mask

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute loss for every data point
        _loss = -K.sum(y_true * K.log(y_pred), axis=-1)

        def _get_real_median(_v):
            """
            given a tensor with shape (batch_size,), compute and return the median

            :param v:
            :return:
            """
            _val = tf.nn.top_k(_v, 33).values
            return 0.5 * (_val[-1] + _val[-2])

        _mean_loss, _var_loss = tf.nn.moments(_loss, axes=[0])
        _median_loss = _get_real_median(_loss)
        _std_loss = tf.sqrt(_var_loss)

        # threshold
        t_l = _median_loss + _l*_std_loss

        _mask_l = 1 - (K.cast(K.greater(_loss, t_l), 'float32') * K.cast(_y_true_flag, 'float32'))
        _loss = _loss * _mask_l

        return _loss
    return crossentropy_outlier_origin_core

