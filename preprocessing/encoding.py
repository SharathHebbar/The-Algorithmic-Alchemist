

class Encoding:

    def __init__(self, choice):
        self.choice = choice

    
    
    def label_encoding(self):
        # Use python map function
        pass

    
    def one_hot_encoding(self):
        # Use One hot Encoding
        pass

    def perform_encoding(self):
        if self.choice == "LabelEncoding":
            self.label_encoding()
        elif self.choice == "OneHotEncoding":
            self.one_hot_encoding()

            

