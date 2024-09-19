import matplotlib.pyplot as plt
import numpy as np
import time

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv  # noqa (no quality assurance)
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=UserWarning)


def HyperparameterOptimization(name, clf, x_train, y_train):
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, clf)
    alg_name = pipeline.steps[1][0]
    if name == "qda":
        param_grid = {"{}__store_covariance".format(alg_name): [True, False],
                      "{}__tol".format(alg_name): [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]}
    elif name == "gpc":
        param_grid = {"{}__warm_start".format(alg_name): [True, False],
                      "{}__copy_X_train".format(alg_name): [True, False]}
    elif name == "lr":
        param_grid = {"{}__penalty".format(alg_name): ["l1", "l2", "elasticnet", None],
                      "{}__tol".format(alg_name): [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
                      "{}__C".format(alg_name): [1e-2, 1e-1, 1e-0, 1e-1, 1e-2],
                      "{}__fit_intercept".format(alg_name): [True, False],
                      "{}__warm_start".format(alg_name): [True, False],
                      "{}__l1_ratio".format(alg_name): [0.1, 0.3, 0.5, 0.7, 0.9]}
    elif name == "gnb":
        param_grid = {"{}__var_smoothing".format(alg_name): [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    elif name == "knn":
        param_grid = {"{}__n_neighbors".format(alg_name): np.arange(1, 11, 2),
                      "{}__weights".format(alg_name): ["uniform", "distance"],
                      "{}__algorithm".format(alg_name): ["auto"],
                      "{}__leaf_size".format(alg_name): [10, 20, 30, 40, 50],
                      "{}__metric".format(alg_name): ["minkowski", "euclidean", "cityblock"]}
    elif name == "dt":
        param_grid = {"{}__criterion".format(alg_name): ["gini", "entropy", "log_loss"],
                      "{}__splitter".format(alg_name): ["best", "random"],
                      "{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
    elif name == "svm":
        param_grid = {"{}__C".format(alg_name): [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                      "{}__kernel".format(alg_name): ["linear", "poly", "rbf", "sigmoid"],
                      "{}__degree".format(alg_name): [1, 2, 3, 4, 5],
                      "{}__gamma".format(alg_name): ["scale", "auto"],
                      "{}__coef0".format(alg_name): [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      '{}__shrinking'.format(alg_name): [True, False]}
    elif name == "sgd":
        param_grid = {"{}__loss".format(alg_name): ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
                      "{}__penalty".format(alg_name): ["l2", "l1", "elasticnet", None]}
    elif name == "rf":
        param_grid = {"{}__criterion".format(alg_name): ["gini", "entropy", "log_loss"],
                      "{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
    elif name == "et":
        param_grid = {"{}__criterion".format(alg_name): ["gini", "entropy", "log_loss"],
                      "{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
    elif name == "gb":
        param_grid = {"{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
    elif name == "hgb":
        param_grid = {"{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
        

    hgs = HalvingGridSearchCV(pipeline, param_grid=param_grid,
                              factor=3,     # the Proportion of Candidates Selected for Each Subsequent Iteration
                              cv=4,         # 4-Fold Cross Validation
                              refit=True,   # If True, Refit an Estimator Using the Best Parameters
                              random_state=42, n_jobs=-1)

    hgs.fit(x_train, y_train)
    return hgs


def ShowEvaluationResult(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, 
                                                   display_labels=["Normal Weight", "Overweight", "Obesity"], 
                                                   cmap=plt.cm.Blues)

    arr = cm.ravel().reshape(3, 3)
    tp_class0 = arr[0, 0]
    fn_class0 = arr[0, 1] + arr[0, 2]
    fp_class0 = arr[1, 0] + arr[2, 0]
    tn_class0 = arr[1, 1] + arr[1, 2] + arr[2, 1] + arr[2, 2]

    tpr_class0 = tp_class0 / (tp_class0 + fn_class0)
    ppv_class0 = tp_class0 / (tp_class0 + fp_class0)
    f1_class0 = 2 * ppv_class0 * tpr_class0 / (ppv_class0 + tpr_class0)

    print("Class 0 - Normal Weight")
    print(">> TPR: %.4f" % (tpr_class0))
    print(">> PPV: %.4f" % (ppv_class0))
    print(">> F1: %.4f" % (f1_class0))

    tp_class1 = arr[1, 1]
    fn_class1 = arr[1, 0] + arr[1, 2]
    fp_class1 = arr[0, 1] + arr[2, 1]
    tn_class1 = arr[0, 0] + arr[0, 2] + arr[2, 0] + arr[2, 2]

    tpr_class1 = tp_class1 / (tp_class1 + fn_class1)
    ppv_class1 = tp_class1 / (tp_class1 + fp_class1)
    f1_class1 = 2 * ppv_class1 * tpr_class1 / (ppv_class1 + tpr_class1)

    print("\nClass 1 - Overweight")
    print(">> TPR: %.4f" % (tpr_class1))
    print(">> PPV: %.4f" % (ppv_class1))
    print(">> F1: %.4f" % (f1_class1))

    tp_class2 = arr[2, 2]
    fn_class2 = arr[2, 0] + arr[2, 1]
    fp_class2 = arr[0, 2] + arr[1, 2]
    tn_class2 = arr[0, 0] + arr[0, 1] + arr[1, 0] + arr[1, 1]

    tpr_class2 = tp_class2 / (tp_class2 + fn_class2)
    ppv_class2 = tp_class2 / (tp_class2 + fp_class2)
    f1_class2 = 2 * ppv_class2 * tpr_class2 / (ppv_class2 + tpr_class2)

    print("\nClass 2 - Obesity")
    print(">> TPR: %.4f" % (tpr_class2))
    print(">> PPV: %.4f" % (ppv_class2))
    print(">> F1: %.4f" % (f1_class2))

    print("\n>> Macro-Average TPR: %.4f" % ((tpr_class0 + tpr_class1 + tpr_class2) / 3))
    print(">> Macro-Average PPV: %.4f" % ((ppv_class0 + ppv_class1 + ppv_class2) / 3))
    print(">> Macro-Average F1: %.4f" % ((f1_class0 + f1_class1 + f1_class2) / 3))
    print(">> Accuracy: %.4f" % ((arr[0, 0] + arr[1, 1] + arr[2, 2]) / (np.sum(arr))))

    
def ClassifierSelection(idx):
    clf = {"qda": QuadraticDiscriminantAnalysis(),
           "gpc": GaussianProcessClassifier(random_state=42, n_jobs=-1),
           "lr": LogisticRegression(random_state=42, solver="saga"),
           "gnb": GaussianNB(),
           "knn": KNeighborsClassifier(n_jobs=-1),
           "dt": DecisionTreeClassifier(random_state=42),
           "svm": SVC(random_state=42),
           "sgd": SGDClassifier(shuffle=False, n_jobs=-1, random_state=42, early_stopping=True),
           "rf": RandomForestClassifier(n_jobs=-1, random_state=42),
           "et": ExtraTreesClassifier(random_state=42),
           "gb": GradientBoostingClassifier(random_state=42),
           "hgb": HistGradientBoostingClassifier(early_stopping=True, random_state=42)}

    print(">> Classifier Type:", end=' ')
    if idx == 0: name = "qda";    print("QDA (Quadratic Discriminant Analysis)")
    elif idx == 1: name = "gpc";  print("GPC (Gaussian Process Classifier)")
    elif idx == 2: name = "lr";   print("LR (Logistic Regression)")
    elif idx == 3: name = "gnb";  print("GNB (Gaussian Naive Bayes)")
    elif idx == 4: name = "knn";  print("kNN (k-Nearest Neighbors)")
    elif idx == 5: name = "dt";   print("DT (Decision Tree)")
    elif idx == 6: name = "svm";  print("SVM (Support Vector Machine)")
    elif idx == 7: name = "sgd";  print("SGD (Stochastic Gradient Descent)")
    elif idx == 8: name = "rf";   print("RF (Random Forest)")
    elif idx == 9: name = "et";   print("ET (Extra Trees)")
    elif idx == 10: name = "gb";  print("GB (Gradient Boosting)")
    elif idx == 11: name = "hgb"; print("HGB (Histogram-Based Gradient Boosting)")
    
    return name, clf[name]
    
    
def SamplingSelection(idx):
    sampling = {"cc": ClusterCentroids(random_state=42),
                "cnn": CondensedNearestNeighbour(random_state=42, n_jobs=-1),
                "enn": EditedNearestNeighbours(n_jobs=-1),
                "renn": RepeatedEditedNearestNeighbours(n_jobs=-1),
                "aknn": AllKNN(n_jobs=-1),
                "nm": NearMiss(n_jobs=-1),
                "ncr": NeighbourhoodCleaningRule(n_jobs=-1),
                "oss": OneSidedSelection(random_state=42, n_jobs=-1),
                "rus": RandomUnderSampler(random_state=42),
                "tl": TomekLinks(n_jobs=-1),
                "ros": RandomOverSampler(random_state=42),
                "smote": SMOTE(random_state=42),
                "adasyn": ADASYN(random_state=42),
                "bsmote": BorderlineSMOTE(random_state=42),
                "svmsmote": SVMSMOTE(random_state=42)}

    print(">> Sampling Type:", end=' ')
    if idx == 1: name = "cc";          print("(U) Cluster Centroids");
    elif idx == 2: name = "cnn";       print("(U) Condensed Nearest Neighbour");
    elif idx == 3: name = "enn";       print("(U) Edited Nearest Neighbours");
    elif idx == 4: name = "renn";      print("(U) Repeated Edited Nearest Neighbours");
    elif idx == 5: name = "aknn";      print("(U) AllKNN");
    elif idx == 6: name = "nm";        print("(U) NearMiss");
    elif idx == 7: name = "ncr";       print("(U) Neighbourhood Cleaning Rule");
    elif idx == 8: name = "oss";       print("(U) One-Sided Selection");
    elif idx == 9: name = "rus";       print("(U) Random Undersampling");
    elif idx == 10: name = "tl";       print("(U) Tomek's Links");
    elif idx == 11: name = "ros";      print("(O) Naive Random Oversampling");
    elif idx == 12: name = "smote";    print("(O) SOMTE (Synthetic Minority Oversampling Technique)");
    elif idx == 13: name = "adasyn";   print("(O) ADASYN (Adaptive Synthetic)");
    elif idx == 14: name = "bsmote";   print("(O) Borderline-SMOTE");
    elif idx == 15: name = "svmsmote"; print("(O) SVM SMOTE");
    
    return sampling[name]
    
    
if __name__ == "__main__":
    # feature = np.load("A_Features.npz")
    # feature = np.load("S_Features.npz")
    feature = np.load("AnS_Features.npz")

    x = feature['x']
    y = feature['y']
    # pid = feature["pid"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    # Control Parameters
    s_type = 13  # Type of Under- and Oversampling Techniques [0, 16]
    c_type = 0  # Type of Classifier [0, 11]

    if s_type == 0:
        print(">> Sampling Type: Nothing")
    else: 
        strategy = SamplingSelection(s_type)
        start_time = time.time()
        x_train, y_train = strategy.fit_resample(x_train, y_train)
        end_time = time.time()
        print("Execution Time (t1):", end_time - start_time)


    c_name, c_clf = ClassifierSelection(c_type)
    start_time = time.process_time()
    model = HyperparameterOptimization(c_name, c_clf, x_train, y_train)
    end_time = time.process_time()
    print("Execution Time (t2):", end_time - start_time)
    # print(model.best_params_)

    y_pred = model.predict(x_test)

    ShowEvaluationResult(y_test, y_pred)
