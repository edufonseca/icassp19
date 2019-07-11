
import numpy as np


def get_accuracy(actual=None, predicted=None):
    """Computes accuracy, done with strings"""
    if predicted == actual:
        return 1.0
    else:
        return 0.0


class Evaluator (object):

    def __init__(self, gt=None, predictions=None, list_labels=None, params_ctrl=None, params_files=None):
        self.gt = gt
        self.predictions = predictions
        self.list_labels = list_labels
        self.train_data = params_ctrl['train_data']

    def evaluate_acc(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        """
        print('\n=====Evaluating ACCURACY - MICRO on the {0} subset of the training set============================='.
              format(self.train_data))
        acc = {}
        for index, row in self.predictions.iterrows():
            pred_per_file = row['label']
            temp = self.gt.loc[self.gt['fname'] == row['fname']]
            for idx_gt, row_gt in temp.iterrows():
                acc[row_gt['fname']] = get_accuracy(actual=row_gt['label'], predicted=pred_per_file)

        sum_acc = 0
        for f_name, score in acc.items():
            sum_acc += score
        self.mean_acc = (sum_acc / len(acc))*100
        print('Number of files evaluated: %d' % len(acc))
        print('Mean Accuracy for files evaluated: %5.2f' % self.mean_acc)

    def evaluate_acc_classwise(self):
        """
        input two dataframes to compare
        :param gt:
        :param predictions:
        :return:
        """
        print('\n=====Evaluating ACCURACY - PER CLASS ======================================================')
        scores = {key: {'nb_files': 0, 'acc_cum': 0} for key in self.list_labels}

        for idx_gt, row_gt in self.gt.iterrows():
            predicted_match = self.predictions.loc[self.predictions['fname'] == row_gt['fname']]
            for idx_pred, row_pred in predicted_match.iterrows():
                pred_per_file = row_pred['label']
                scores[row_gt['label']]['nb_files'] += 1
                # computing ACCURACY and saving it in the due class
                scores[row_gt['label']]['acc_cum'] += get_accuracy(actual=row_gt['label'], predicted=pred_per_file)

        total = 0
        perclass_acc = []
        for label, v in scores.items():
            # If encounter 0 accuracy, don't want program to crash on divide by zero
            if v['nb_files'] == 0:
                mean_acc = 0
            else:
                mean_acc = (v['acc_cum'] / v['nb_files'])*100
            print('%-21s | number of files in total: %-4d | Accuracy: %6.3f' % (label, v['nb_files'], mean_acc))
            perclass_acc.append(mean_acc)
            total += v['nb_files']
        print('Total number of files: %d' % total)

        print('\n=====Printing sorted classes for ACCURACY - PER CLASS ========================================')
        perclass_acc_np = np.array(perclass_acc)
        idx_sort = np.argsort(-perclass_acc_np)
        for i in range(len(self.list_labels)):
            print('%-21s | number of files in total: %-4d | Accuracy: %6.3f' %
                  (self.list_labels[idx_sort[i]], scores[self.list_labels[idx_sort[i]]]['nb_files'],
                   perclass_acc[idx_sort[i]]))

    def print_summary_eval(self):

        print('\n=====================================================================================================')
        print('=====================================================================================================')
        print('SUMMARY of evaluation:')
        print('Mean Accuracy for files evaluated: %5.2f' % self.mean_acc)
        print('\n=====================================================================================================')

