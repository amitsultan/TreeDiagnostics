import pandas as pd
import numpy as np
import pickle
from TreeDiagnostics import *


df_all = pd.read_csv("cardiotocography3clases.csv")
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

before = df_all[0:1500]
after = df_all[1500:]

max = before.max(axis=0)
min = before.min(axis=0)
avg = before.mean(axis=0)
std = before.std(axis=0)

statistics = pd.concat([max, min, avg, std],axis=1)
statistics.columns = ["max", "min", "avg", "std"]

with open("cardiotocography3clases_TREE", "rb") as file:
    tree_model = pickle.load(file)

print(tree_model.tree_.feature)

class_name = "clase"
features = list(after.columns)
features.remove(class_name)

y_test = after[class_name]
x_test = after[features]


std_tree_diagnostic = TreeDiagnostics(normalize_method=NormOptions.STD_NORM)
minmax_tree_diagnostic = TreeDiagnostics(normalize_method=NormOptions.MINMAX_NORM)


matrix = std_tree_diagnostic.diagnose_tree(clf=tree_model, statistics=statistics, data_x=x_test, data_y=y_test)
print(matrix)