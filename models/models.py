from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    SGDRegressor,
    
) 
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
) 
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.naive_bayes import GaussianNB

from xgboost import (
    XGBClassifier,
    XGBRegressor
)
from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)

from sklearn.cluster import KMeans


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class Models():

    def __init__(self, task_name, model_name, xtrain, xtest, ytrain, ytest):
        self.task_name = task_name
        self.model_name = model_name
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def model_initialization(self):

        if self.task_name == 'Classification':
            if self.model_name == 'RandomForest':
                self.model = RandomForestClassifier()
            elif self.model_name == 'DecisionTree':
                self.model = DecisionTreeClassifier()
            elif self.model_name == 'XGBoost':
                self.model = XGBClassifier()
            elif self.model_name == 'CatBoost':
                self.model = CatBoostClassifier
            elif self.model_name == 'Naive Bayes':
                self.model = GaussianNB()
            elif self.model_name == 'Logistic Regression':
                self.model = LogisticRegression()
            elif self.model_name == 'Support Vector Machines':
                self.model = SVC()
            else:
                pass

        elif self.task_name == "Regression":
            if self.model_name == 'RandomForest':
                self.model = RandomForestRegressor()
            elif self.model_name == 'DecisionTree':
                self.model = DecisionTreeRegressor()
            elif self.model_name == 'XGBoost':
                self.model = XGBRegressor()
            elif self.model_name == 'CatBoost':
                self.model = CatBoostRegressor
            elif self.model_name == 'Naive Bayes':
                self.model = GaussianNB()
            elif self.model_name == 'Linear Regression':
                self.model = LinearRegression()
            elif self.model_name == 'Support Vector Machines':
                self.model = SVR()
            elif self.model_name == "ElasticNet":
                self.model = ElasticNet()
            elif self.model_name == "Stochastic Gradient Descent":
                self.model = SGDRegressor()
            else:
                pass
        
        elif self.task_name == "Clustering":
            if self.model_name == 'KMeans':
                self.model = KMeans()
        # return self.model
    
    def fit_model(self):
        self.model.fit(self.xtrain, self.ytrain)
        self.ytrain_pred = self.model.predict(self.xtrain)
        self.ytest_pred = self.model.predict(self.xtest)

        self.train_accuracy = accuracy_score(self.ytrain, self.ytrain_pred)
        self.test_accuracy = accuracy_score(self.ytest, self.ytest_pred)

        self.train_precision = precision_score(self.ytrain, self.ytrain_pred)
        self.test_precision = precision_score(self.ytest, self.ytest_pred)

        self.train_recall = recall_score(self.ytrain, self.ytrain_pred)
        self.test_recall = recall_score(self.ytest, self.ytest_pred)

        self.train_f1_score = f1_score(self.ytrain, self.ytrain_pred)
        self.test_f1_score = f1_score(self.ytest, self.ytest_pred)

        self.train_classification_report = classification_report(self.ytrain, self.ytrain_pred)
        self.test_classification_report = classification_report(self.ytest, self.ytest_pred)

        self.train_confusion_matrix = confusion_matrix(self.ytrain, self.ytrain_pred)
        self.test_confusion_matrix = confusion_matrix(self.ytest, self.ytest_pred)

        return (
            (self.train_accuracy, self.test_accuracy),
            (self.train_precision, self.test_precision),
            (self.train_recall, self.test_recall),
            (self.train_f1_score, self.test_f1_score),
            (self.train_classification_report, self.test_classification_report),
            (self.train_confusion_matrix, self.test_confusion_matrix)
        ) 
            
