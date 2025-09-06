import numpy as np

def display_cm(cm, labels, hide_zeros=False, display_metrics=False):
    """Display confusion matrix with labels, along with
       metrics such as Recall, Precision and F1 score.
       Based on Zach Guo's print_cm gist at
       https://gist.github.com/zachguo/10296432
    """

    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0

    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0,1))

    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth

    # Print header
    header = "    Pred " + " ".join([f"{label:>{columnwidth}}" for label in labels]) + f"{'Total':>{columnwidth}}"
    print(header)
    print("    True")

    # Print rows
    for i, label1 in enumerate(labels):
        row = f"    {label1:>{columnwidth}}"
        for j in range(len(labels)):
            cell = f"{cm[i, j]:{columnwidth}d}"
            if hide_zeros and cm[i, j] == 0:
                cell = empty_cell
            row += cell
        row += f"{sum(cm[i, :]):{columnwidth}d}"
        print(row)

    # Print metrics
    if display_metrics:
        print("\nPrecision", " ".join([f"{p:{columnwidth}.2f}" for p in precision]), f"{total_precision:{columnwidth}.2f}")
        print("   Recall", " ".join([f"{r:{columnwidth}.2f}" for r in recall]), f"{total_recall:{columnwidth}.2f}")
        print("       F1", " ".join([f"{f:{columnwidth}.2f}" for f in F1]), f"{total_F1:{columnwidth}.2f}")


def display_adj_cm(cm, labels, adjacent_facies, hide_zeros=False, display_metrics=False):
    """Display confusion matrix that counts adjacent facies as correct."""
    adj_cm = np.copy(cm)

    for i in np.arange(0, cm.shape[0]):
        for j in adjacent_facies[i]:
            adj_cm[i][i] += adj_cm[i][j]
            adj_cm[i][j] = 0.0

    display_cm(adj_cm, labels, hide_zeros, display_metrics)
