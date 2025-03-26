#!/usr/bin/env python3
from sklearn.metrics import classification_report

def print_confusion_matrix(y_true, y_pred, target_names):
    """Prints the confusion matrix."""
    print('Confusion matrix:')
    print(classification_report(y_true, y_pred, target_names))
    print()
    return None