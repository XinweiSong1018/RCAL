'''
* @name: metric.py
* @description: Evaluation metrics for multimodal sentiment analysis tasks.
*               Note: This file is adapted from MMSA (https://github.com/thuiar/MMSA/tree/master).
'''

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Expose only the MetricsTop class for external use
__all__ = ['MetricsTop']


class MetricsTop():
    """
    Main metric evaluation class for different multimodal sentiment datasets.
    Supports MOSI, MOSEI, and CH-SIMS regression tasks.
    """
    def __init__(self):
        # Mapping from dataset name to its specific evaluation function
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'MOSEI': self.__eval_mosei_regression,
            'CHSIMS': self.__eval_sims_regression
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute accuracy by rounding predictions and comparing with ground truth labels.

        Args:
            y_pred (array): Predicted values
            y_true (array): Ground truth labels

        Returns:
            float: Accuracy score
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        """
        Evaluation function for MOSEI and MOSI (continuous sentiment from -3 to +3).

        Metrics:
            - Binary accuracy/F1 with and without zero
            - Multiclass accuracy after clipping into 7, 5, 3 classes
            - Mean Absolute Error (MAE)
            - Pearson correlation

        Returns:
            dict: All evaluation scores
        """
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        # Clip predictions to various scales
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        # Core regression metrics
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        # Binary metrics ignoring zero
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_preds, non_zeros_binary_truth, average='weighted')

        # Binary metrics including zero
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_preds, binary_truth, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results


    def __eval_mosi_regression(self, y_pred, y_true):
        """
        Alias for MOSEI evaluation logic since metrics are the same.

        Args:
            y_pred: Model predictions
            y_true: Ground truth

        Returns:
            dict: Evaluation results
        """
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        """
        Evaluation function for CH-SIMS (continuous sentiment from -1 to 1).

        Discretizes regression output into 2, 3, and 5 classes to evaluate classification-style performance.

        Returns:
            dict: Evaluation results
        """
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        # Metrics
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_preds_a2, test_truth_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr, # Correlation Coefficient
        }
        return eval_results
    
    def getMetics(self, datasetName):
        """
        Retrieve the metric function based on dataset name.

        Args:
            datasetName (str): One of ['MOSI', 'MOSEI', 'CHSIMS']

        Returns:
            Callable: Corresponding evaluation function
        """
        return self.metrics_dict[datasetName.upper()]
