
def task_names():
    return ['Classification', 'Regression', 'Clustering']


def model_lists(task_name):
    if task_name == 'Classification': 
        return [
            "RandomForest",
            "DecisionTree",
            "XGBoost",
            "CatBoost",
            "Naive Bayes",
            "Logistic Regression",
            "Support Vector Machines"
        ]
    elif task_name == "Regression":
        return [
            "Linear Regression",
            "ElasticNet",
            "Stochastic Gradient Descent",
            "RandomForest",
            "DecisionTree",
            "XGBoost",
            "CatBoost",
            
            "Support Vector Machines",
            
        ]
    elif task_name == "Clustering":
        return [
            "KMeans",
            "Hierarchical"
        ]
    else:
        return [None]
    

def transformation_list():
    return ['Standardization', "Normalization"]

def encoding_list():
    return ['LabelEncoding', "OneHotEncoding"]