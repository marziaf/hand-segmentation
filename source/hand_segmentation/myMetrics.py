

# TODO
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import metrics

def compute_and_print_IoU_per_class(confusion_matrix, num_classes, class_mask=None):
    """
    Computes and prints mean intersection over union divided per class
    :param confusion_matrix: confusion matrix needed for the computation
    """
    mIoU = 0
    mIoU_nobackgroud = 0
    IoU_per_class = np.zeros([num_classes], np.float32)
    true_classes = 0

    per_class_pixel_acc = np.zeros([num_classes], np.float32)

    mean_class_acc_num = 0

    true_classes_pix = 0
    mean_class_acc_den = 0

    mean_class_acc_num_nobgr = 0
    mean_class_acc_den_nobgr = 0
    mean_class_acc_sum_nobgr = 0
    mean_class_acc_sum = 0

    if class_mask == None:
        class_mask = np.ones([num_classes], np.int8)

    for i in range(num_classes):

        if class_mask[i] == 1:
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            # TN = np.sum(confusion_matrix) - TP - FP - FN

            denominator = (TP + FP + FN)
            # If the denominator is 0, we need to ignore the class.
            if denominator == 0:
                denominator = 1
            else:
                true_classes += 1

            # per-class pixel accuracy
            if not TP == 0:
                # if not np.isnan(TP):
                tmp = (TP + FN)
                per_class_pixel_acc[i] = TP / tmp

            IoU = TP / denominator
            IoU_per_class[i] += IoU
            mIoU += IoU

            if i > 0:
                mIoU_nobackgroud += IoU

            # mean class accuracy
            if not np.isnan(per_class_pixel_acc[i]):
                mean_class_acc_num += TP
                mean_class_acc_den += TP + FN

                mean_class_acc_sum += per_class_pixel_acc[i]
                true_classes_pix += 1

                if i > 0:
                    mean_class_acc_num_nobgr += TP
                    mean_class_acc_den_nobgr += TP + FN
                    mean_class_acc_sum_nobgr += per_class_pixel_acc[i]

    mIoU = mIoU / true_classes
    mIoU_nobackgroud = mIoU_nobackgroud / (true_classes - 1)

    mean_pix_acc = mean_class_acc_num / mean_class_acc_den
    mean_pixel_acc_nobackground = mean_class_acc_num_nobgr / mean_class_acc_den_nobgr

    print("---------------------------------------------------------------------------")
    for k in range(0, num_classes):
        if IoU_per_class[k] > 0:
            print("IoU for class " + str(k) + " :" + str(IoU_per_class[k] * 100))
            print("Pixel Acc for class " + str(k) + " :" + str(per_class_pixel_acc[k] * 100))
            print("---------------------------------------------------------------------------")
    print(" ")
    print("--METRICS--")
    print(' mean_class_acc :' + str((mean_class_acc_sum / true_classes_pix) * 100))
    print(' mean pix acc :' + str(mean_pix_acc * 100))
    print(' mean_pixel_acc_no_background :' + str(mean_pixel_acc_nobackground * 100))
    print(" mIoU:" + str(mIoU * 100))
    print(" mIoU_nobackgroud:" + str(mIoU_nobackgroud * 100))
    print("---------------------------------------------------------------------------")

    return mIoU * 100