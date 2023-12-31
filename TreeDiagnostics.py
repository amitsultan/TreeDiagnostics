import copy
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import BaseDecisionTree

from helper import NormOptions, invoke_normalization


class TreeDiagnostics:
    """
    TreeDiagnostics helps in diagnosing a decision tree by investigating the individual decisions made in the nodes.
    E.G. For a given example and a path p_0, to p_n, it calculates alternates path with alternates thresholds and
    provide information about the new threshold diff compared to the old threshold. The new threshold is set by
    taking the example feature value and adding/subtracting epsilon of it.
    Attributes
    ----------
    norm_opt : NormOptions
        The normalization method to be used.
        can be:
            NO_ACTION = 1
            STD_NORM = 2
            MINMAX_NORM = 3
    norm_func : callable
        The function to invoke normalization.
    """

    def __init__(self, normalize_method: NormOptions, delete_leaves_nodes: bool = False) -> None:
        """
        Initialize the TreeDiagnostics object.
        Parameters
        ----------
        normalize_method : NormOptions
            The normalization method to be used.
        delete_leaves_nodes: Bool
            Delete leaves columns from the output, otherwise - fill with -inf.
        """
        self.norm_opt = normalize_method
        self.norm_func = invoke_normalization
        self.delete_leaves = delete_leaves_nodes

    def diagnose_tree(self, clf: BaseDecisionTree, statistics: pd.DataFrame, data_x: pd.DataFrame,
                      data_y: pd.Series) -> pd.DataFrame:
        """
        Diagnose a decision tree and returns a dataframe with the diagnosis results.
        Rows - Instances
        Columns - Tree Nodes
        -Inf are leaf nodes.
        Parameters
        ----------
        clf : BaseDecisionTree
            The classifier to be diagnosed.
        statistics : pd.DataFrame
            The statistics data.
        data_x : pd.DataFrame
            The input data.
        data_y : pd.Series
            The output data.

        Returns
        ----------
        pd.DataFrame
            The dataframe with diagnosis results.
        """
        is_correctly_classified = []
        masked_output, decision_nodes_indices, leaf_indices = self.__prepare_output_masked(clf, data_x)
        for i in range(len(masked_output)):
            instance = masked_output[i]
            example_x, example_y = data_x.iloc[i].values, data_y.iloc[i]
            is_correct = example_y == clf.predict(example_x.reshape(1, -1))
            is_error = [not is_correct[0]]
            is_correctly_classified.append(is_error)
            visited_nodes = np.where(instance == 1)[0]
            for node_id in visited_nodes:
                is_correct_new, new_threshold, new_clf = self.__step(
                    clf=clf, features_statistics=statistics, node_id=node_id, example=example_x, true_ground=example_y
                )
                # if corr -> corr 0 miss -> miss = 0
                if (is_correct and is_correct_new) or (not is_correct and not is_correct_new):
                    masked_output[i][node_id] = 0
                # if miss -> correct = thresh corr -> miss thresh
                elif (is_correct and not is_correct_new) or (not is_correct and is_correct_new):
                    masked_output[i][node_id] = new_threshold
        results = np.hstack((masked_output, is_correctly_classified))
        df = pd.DataFrame(results)
        if self.delete_leaves:
            df = df.drop(columns=leaf_indices, axis=1)
        return df

    def __prepare_output_masked(self, clf: BaseDecisionTree, data_x: pd.DataFrame) -> Tuple:
        """
        Prepare masked output for diagnosis. It prepares the layout
        for the output matrix. (Rows - instances, Columns - Tree Nodes).
        with the following values:
            1 - traveled node.
            0 - Not traveled node.
            -inf - leaves.
        Parameters
        ----------
        clf : BaseDecisionTree
            The classifier to be diagnosed.
        data_x : pd.DataFrame
            The input data.

        Returns
        ----------
        np.ndarray
            The masked output.
        """
        arr = clf.decision_path(data_x).toarray().astype('float64')
        decision_nodes_indices = np.where(clf.tree_.feature != -2)[0]
        leaf_indices = np.where(clf.tree_.feature == -2)[0]
        arr[:, leaf_indices] = -float('inf')
        return arr, decision_nodes_indices, leaf_indices

    def __step(self, clf: BaseDecisionTree, features_statistics: pd.DataFrame, node_id: int, example: np.ndarray,
               true_ground: int) -> Tuple[bool, float, BaseDecisionTree]:
        """
        Performs one step of diagnosis.
        Parameters
        ----------
        clf : BaseDecisionTree
            The classifier to be diagnosed.
        features_statistics : pd.DataFrame
            The statistics data.
        node_id : int
            The node id in the decision tree.
        example : np.ndarray
            An example from the input data.
        true_ground : int
            The ground truth value for the example.

        Returns
        ----------
        Tuple[bool, float, BaseDecisionTree]
            A tuple consisting of a boolean verdict, a new threshold, and a new classifier.
        """
        clf = copy.deepcopy(clf)  # Can be removed if carefully managed
        tree = clf.tree_
        feature = tree.feature[node_id]
        statistics = features_statistics.iloc[feature]
        threshold = tree.threshold[node_id]
        new_threshold = example[feature] - np.finfo(float).eps if example[feature] <= threshold else example[feature] + np.finfo(float).eps
        diff = self.norm_func(np.abs(tree.threshold[node_id] - new_threshold), statistics, self.norm_opt)
        tree.threshold[node_id] = new_threshold
        verdict = clf.predict(example.reshape(1, -1)) == true_ground
        return verdict, diff, clf
