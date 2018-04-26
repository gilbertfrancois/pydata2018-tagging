import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    :param cm:          Confusion matrix
    :param classes:     List of class labels
    :param normalize:   Applies normalization if set to True. Default is False
    :param title:       Title of the plot
    :param cmap:        Color map. Default is Blues
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_y_proba(y_pred, dict_classes, threshold):
    """
    Plots a bar plot of probability values per class.
    :param y_pred:          Vector with predicted probabilities.
    :param dict_classes:    Dictionary with class labels
    :param threshold:       Threshold
    :return:
    """
    ind_x = np.arange(8)
    y_thr = np.ones(len(ind_x)) * threshold
    plt.figure()
    plt.bar(ind_x, y_pred, 0.9, color='black')
    plt.ylim([0,1])
    plt.plot(ind_x, y_thr, label='threshold', color='black', alpha=0.3)
    plt.title('Probability per class')
    plt.legend()
    plt.xticks(ind_x)
    plt.gca().set_xticklabels(dict_classes.values(), rotation=60)
    plt.tight_layout()


def show_pred_proba(y_pred_list, doc_list, dict_classes, threshold=None):
    """
    Shows the predicted probability vector and renders the probabilities in a bar plot.
    :param y_pred_list:     List of predicted probabilities
    :param doc_list:        List of input documents
    :param threshold:       Threshold, 0 ≤ x ≤ 1. When None, the threshold will be estimated per document.
    :return:
    """
    for idx1 in range(len(y_pred_list)):
        # Compute the threshold based on the max value of the probability vector,
        # when no fixed value is given.
        if threshold is None:
            _threshold = np.max(y_pred_list[idx1]) / 4
        else:
            _threshold = threshold
        print('threshold: {:.3f}'.format(_threshold))
        print(doc_list[idx1], '\n')
        print('y_pred: {}'.format(y_pred_list[idx1]))
        plot_y_proba(y_pred_list[idx1], dict_classes, _threshold)
        for idx2, y_i in enumerate(y_pred_list[idx1]):
            if (y_i > _threshold):
                print('* {}'.format(dict_classes[idx2 + 1]))
