import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd


from lists import task_names, model_lists, transformation_list, encoding_list
from models.models import Models

class UI():

    def __init__(self):
        st.header("Classical ML App")
        tasks = task_names()

        self.task_name = st.sidebar.selectbox(
            label="Choose the task you want to perform",
            options = tasks,
            index=None,
            help="Choose your preferred tasks"
        )

        if self.task_name is None:
            model_selection_disablity = True
        else:
            model_selection_disablity = False

        model_selection_lists = model_lists(self.task_name)

        self.model_name = st.sidebar.selectbox(
            label="Choose your Model",
            options=model_selection_lists,
            disabled=model_selection_disablity,
            index=None,
            help="Choose your model"
        )

        if self.model_name is None:
            transformation_selection_disablity = True
            split_disablity = True
            encoding_disablity = True
        else:
            transformation_selection_disablity = False
            split_disablity = False
            encoding_disablity = False

        self.transformation = st.sidebar.selectbox(
            label="Choose your transformation",
            options=transformation_list(),
            disabled=transformation_selection_disablity,
            index=None,
            help="Choose your transformation"
        )


        self.split_ratio = st.sidebar.slider(
            label="Choose your Split size",
            min_value=0.1,
            max_value=0.5,
            step=0.05,
            value=0.1,
            disabled=split_disablity,
            help="Choose your transformation"
        )


        self.encoding = st.sidebar.selectbox(
            label="Choose your Encoding",
            options=encoding_list(),
            disabled=encoding_disablity,
            index=None,
            help="Choose your transformation"
        )


        self.filename = st.file_uploader(
            label="Upload an CSV File",
            type="csv",
            help="Upload Your CSV File"
        )

    
        st.subheader("Output Area")

    def button_init(self):
        if self.filename is not None and self.df is not None:
            if self.task_name != "Clustering":
                if self.target is None:
                    self.button_disablity = True
                else:
                    self.button_disablity = False
            elif self.task_name == "Clustering":
                self.button_disablity = False
            else:
                self.button_disablity = True
        else:
            self.button_disablity = True

        
        self.button = st.button(
            label="Submit",
            disabled=self.button_disablity,
            help="Click on submit to train a model"
        )
    

    def split(self):
        
        y = self.df[self.target]
        
        x = self.df.drop([self.target], axis=1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
                x,
                y,
                test_size=self.split_ratio,
                stratify=self.df[self.target]
            )



    def train(self):
        if self.model_name is not None and self.split_ratio is not None and self.filename is not None:
            self.df = pd.read_csv(self.filename)
            cols = self.df.columns
            if self.task_name != "Clustering":
                self.target = st.sidebar.selectbox(
                    label="Select Target Variable",
                    options=cols,
                    index=None,
                    help="Select target variable for prediction"
                )
            self.button_init()
            st.write("Data Head")
            st.write(self.df.head())
            st.write("Data Describe")
            st.write(self.df.describe())
            
            if self.button:
                self.split()
                models = Models(self.task_name, self.model_name, self.xtrain, self.xtest, self.ytrain, self.ytest)
                models.model_initialization()
                ((self.train_accuracy, self.test_accuracy),
                (self.train_precision, self.test_precision),
                (self.train_recall, self.test_recall),
                (self.train_f1_score, self.test_f1_score),
                (self.train_classification_report, self.test_classification_report),
                (self.train_confusion_matrix, self.test_confusion_matrix)) = models.fit_model()
                # st.write(models.fit_model())

                result = {
                    'Train Accuracy': self.train_accuracy,
                    'Test Accuracy': self.test_accuracy,
                    'Train Precision': self.train_precision,
                    'Test Precision': self.test_precision,
                    'Train Recall': self.train_recall,
                    'Test Recall': self.test_recall,
                    'Train F-Score': self.train_f1_score,
                    'Test F-Score': self.test_f1_score,
                }

                result_df = pd.DataFrame(result.values(), index=result.keys(), columns=['Value'])
                result_df.index.name = 'Metric'
                st.write(result_df)

ui = UI()
ui.train()