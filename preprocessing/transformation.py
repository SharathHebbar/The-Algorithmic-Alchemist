from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Transformation():

    def __init__(self, choice, xtrain, xtest):
        self.choice = choice
        self.xtrain = xtrain
        self.xtest = xtest

    def perform_transformation(self):
        # Further add TFIDF Count Vectorizer
        if self.choice == "Standardization":
            self.scaler = StandardScaler()

        elif self.choice == "Normalization":
           self.scaler = MinMaxScaler()
        
        xtrain_transformed = self.scaler.fit_transform(self.xtrain)
        xtest_transformed = self.scaler.transform(self.xtest)

        return xtrain_transformed, xtest_transformed

        