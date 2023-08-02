"# TreeDiagnostics" 

# main.py

This file contains a sampe made by Shaked for the pipeline. It contains the loading of the
dataset, statistics of features, and model initialization. <br>

## model init
TreeDiagnostics(normalize_method=NormOptions.NO_ACTION, delete_leaves_nodes=True)
* normalize_method - Can be one of the following:
  * NO_ACTION - No normalization.
  * STD_NORM - Perform Standard scaler normalization.
  * MINMAX_NORM - Perform min-max normalization.
* delete_leaves_nodes - Whether to delete leaves nodes out of the output matrix. 
##### Below is a call to our tree_diagnostic model which analysis tree_model object with the given statistics and data.<br>
matrix_std = std_tree_diagnostic.diagnose_tree(clf=tree_model, statistics=statistics, data_x=x_test, data_y=y_test)<br>
matrix_minmax = minmax_tree_diagnostic.diagnose_tree(clf=tree_model, statistics=statistics, data_x=x_test, data_y=y_test)

To run the algorithm diagnostic you need to pass the following parameters:
* clf - Tree Classifier (SK-learn object) BaseDecisionTree.
* statistics - Feature statistics with the feature ordered in the same order of the tree features.
* data_x - feature data for the test set.
* data_y - label data for the test set (is_correctly classified).

The output is a matrix where columns are tree nodes, and rows are instances.
It has an additional column is_correctly_classified, containing true if the tree did classify the instance correctly.