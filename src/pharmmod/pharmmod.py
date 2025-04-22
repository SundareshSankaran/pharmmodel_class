class PharmMod:
    """This class helps you initialise & perform operations on a Python class object called PharmMod."""
    def __init__(self, path = None, name=None, creationTimeStamp=None,  modifiedTimeStamp=None, 
                 createdBy=None, modifiedBy=None, dataframes=[], profiles=[], profile_reports=[], 
                 target=None, models_run=[]) -> None:
        
        import datetime

        # Initialisation of attributes
        self.path = None
        self.name = None
        self.creationTimeStamp = None
        self.modifiedTimeStamp = None
        self.createdBy = None
        self.modifiedBy = None
        self.dataframes = []
        self.profiles = []
        self.profile_reports = []
        self.target = None
        self.models_run = []

        # Assign attributes based on what's passed
        self.path = path if path else self.path
        self.name = name if name else self.name
        self.creationTimeStamp = creationTimeStamp if creationTimeStamp else datetime.datetime.now()
        self.modifiedTimeStamp = modifiedTimeStamp if modifiedTimeStamp else datetime.datetime.now()
        self.createdBy = createdBy if createdBy else self.createdBy
        self.modifiedBy = modifiedBy if modifiedBy else self.modifiedBy
        self.dataframes = dataframes if dataframes else self.dataframes
        self.profiles = profiles if profiles else self.profiles
        self.profile_reports = profile_reports if profile_reports else self.profile_reports
        if self.path:
            self.read_data(self.path)
        self.target = target if target else self.target
        self.models_run = models_run if models_run else self.models_run

    def path_check(self, path):
        """This function checks if the path provided is a directory or a file, extension of a file and if it exists"""
        import os
        if not os.path.exists(path):
            return {"exists":False, "is_dir":False, "is_file":False, "extn":None, "dir":None, "filename":None}
        else:
            if os.path.isdir(path):
                return {"exists":True, "is_dir":True, "is_file":False, "extn":None,"dir":path, "filename":None}
            elif os.path.isfile(self.path):
                dir_name = os.path.dirname(path)
                filename, extn = os.path.splitext(path)
                return {"exists":True, "is_dir":False, "is_file":True, "extn":extn,"dir":dir_name,"filename":filename}
            
    def read_data(self, path=None):
        """This function reads data from the path provided and returns a pandas dataframe"""
        import pandas as pd
        import os
        if not path:
            path = self.path
        filename, extn = os.path.splitext(os.path.realpath(path))
        self.name = filename
        if extn == ".xlsx":
            pd = pd.read_excel(path)
        elif extn == ".csv":
            pd = pd.read_csv(path)
        elif extn == ".json":
            pd = pd.read_json(path)
        elif extn == ".txt":
            pd = pd.read_csv(path, sep="\t")
        elif extn == ".tab":
            pd = pd.read_csv(path, sep="\t")
        else:
            print("Unsupported file format")
            return None
        if len(self.dataframes) == 0:
                self.dataframes.append(pd) 
        else:
                self.dataframes[0] = pd
        return pd
    
    def describe(self, dataframe_indicator=0):
        """This function returns a dataframe profile"""
        import json
        import os
        profile = self.dataframes[dataframe_indicator].describe().T
        profile["column_name"]=profile.index
        profile["id"]=list(range(0,len(profile.index)))
        profile.set_index("id", inplace=True)
        if len(self.profiles) == 0:
            self.profiles.append(profile)
        else:
            self.profiles[dataframe_indicator] = profile
        from ydata_profiling import ProfileReport
        profile_report = ProfileReport(self.dataframes[dataframe_indicator], title=f"Profiling Report for {self.name}", explorative=True)
        profile_report.to_file(f"{self.name}_profile_report.html")
        profile_report_json = profile_report.to_json()
        if( len(self.profile_reports) == 0):
            self.profile_reports.append(json.loads(profile_report_json))
        else:
            self.profile_reports[dataframe_indicator] = json.loads(profile_report_json)
        return profile
 
    def visualize(self, partition_indicator=0, col = [], plot_type = "histogram", bins = 10):
        """This function visualizes the data in the dataframe"""
        import matplotlib.pyplot as plt 
        from math import ceil
        df_data = self.dataframes[partition_indicator]
        if len(col) > 0:
            df_data = df_data[col]
        columns = list(df_data.columns)
        fig, axs = plt.subplots(ncols=2, nrows=ceil(len(columns)/2), figsize=(25, 12))
        axs = axs.ravel()        
        for i in range(len(columns)):
            ax=axs[i]
            if df_data[columns[i]].dtype in ['float64', 'int64']:
                ax.hist(df_data[columns[i]], bins=bins, color='blue', alpha=0.4, density=True, label=str(columns[i]))
            else:
                df_data[columns[i]].value_counts(normalize=True).to_frame().\
                    sort_values(by=columns[i]).plot(y='proportion',kind='bar', ax=ax, color='blue', rot=45, label=str(columns[i]), alpha=0.4, position=1, width=0.2)
            ax.legend()
            ax.set_title(columns[i])
        plt.tight_layout()
        plt.show()

    def visualize_category(self, col=[], partition_indicator=0, cardinality_threshold=10):
        """This function visualizes the categorical columns in the dataframe"""
        import matplotlib.pyplot as plt
        df_data = self.dataframes[partition_indicator]
        if len(col) > 0:
            df_data = df_data[col]
        columns = list(df_data.columns)
        for column in columns:
            if df_data[column].nunique() > cardinality_threshold:
                print(f"Column {column} has more than {cardinality_threshold} unique values. Please select a different column.")
                columns.remove(column)
            else:
                labels = df_data[column].value_counts().index
                values = df_data[column].value_counts().values
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
                axs[0].set_title(f'{df_data[column].name} Distribution (Pie Chart)')
                axs[1].bar(labels, values)
                axs[1].set_title(f'{df_data[column].name} Distribution (Bar Chart)')
                axs[1].set_xlabel(df_data[column].name)
                axs[1].set_ylabel('Count')
                plt.tight_layout()
                plt.show()
    
    def corr(self,partition_indicator=0, col = []):
        """This function returns the correlation matrix of the dataframe"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        df_data = self.dataframes[partition_indicator]
        if len(col) > 0:
            df_data = df_data[col]
        corr = df_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
        return corr
    
    def starting_odds(self, partition_indicator=0, target=None):
        """This function returns the starting odds of the target variable in the dataframe"""
        import pandas as pd
        df_data = self.dataframes[partition_indicator]
        odds = pd.concat([df_data[target].value_counts().to_frame(), df_data[target].value_counts(normalize=True).to_frame()], axis=1)
        return odds
    
    def partition(self, val_pct = 0.4, test_pct = 0.1):
        from sklearn.model_selection import train_test_split
        df_data = self.dataframes[0]
        train, test = train_test_split(df_data, train_size=(1 - val_pct - test_pct))
        valid, test =train_test_split(test, train_size=(val_pct/(val_pct + test_pct)))
        if len(self.dataframes) == 1:
            self.dataframes.append(train)
            self.dataframes.append(valid)
            self.dataframes.append(test)
        elif len(self.dataframes) == 2:
            self.dataframes[1] = train
            self.dataframes[2] = valid
            self.dataframes.append(test)
        elif len(self.dataframes) > 2:
            self.dataframes[1] = train
            self.dataframes[2] = valid
            self.dataframes[3] = test

    def label_encode(self, col=[]):
        """This function label encodes the categorical columns in the dataframe"""
        from sklearn.preprocessing import LabelEncoder
        df_data = self.dataframes[0]
        if len(col) == 0:
            col = df_data.select_dtypes(include=['object','int64']).columns
        for i in col:
            le = LabelEncoder()
            df_data[f"{i}_le"] = le.fit_transform(df_data[i])
            for j in range(1, len(self.dataframes)):
                self.dataframes[j][f"{i}_le"] = le.transform(self.dataframes[j][i])
        
    def run_model(self, model, target= None, col=[]):
        """This function runs the model on the training data"""
        from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
        import matplotlib.pyplot as plt
        train = self.dataframes[1]
        valid = self.dataframes[2]
        test = self.dataframes[3]
        if target is None:
            target = self.target
        if len(col) == 0:
            col = train.drop(target, axis=1).columns
        model.fit(train[col], train[target])
        validation_f1_score = round(100*f1_score(valid[target], model.predict(valid[col])),2)
        training_f1_score = round(100*f1_score(train[target], model.predict(train[col])),2)
        test_f1_score = round(100*f1_score(test[target], model.predict(test[col])),2)
        print('Training F1 Score:', training_f1_score)
        print('Validation F1 Score:', validation_f1_score)
        print('Test F1 Score:', round(100*f1_score(test[target], model.predict(test[col])),2))
        fig, axs = plt.subplots(ncols=2, figsize=(16,5))
        disp = ConfusionMatrixDisplay(confusion_matrix(valid[target], model.predict(valid[col]), normalize='true'))
        disp.plot(cmap=plt.cm.Blues, ax=axs[0])
        axs[0].set_title('Validation Confusion Matrix')
        disp = ConfusionMatrixDisplay(confusion_matrix(test[target], model.predict(test[col]), normalize='true'))
        disp.plot(cmap=plt.cm.Blues, ax=axs[1])
        axs[1].set_title('Test Confusion Matrix')
        plt.show()
        return {
            "model_type": "Model Run",
            "model": model,
            "Training F1": training_f1_score,
            "Validation F1": validation_f1_score,
            "Test F1": test_f1_score,
            "target": target,
            "features": col
        }

    def forest_classifier(self, target=None, col=[], model_type = "Random Forest Classifier"):
        """This function fits a random forest classifier to the training data"""
        from sklearn.ensemble import RandomForestClassifier as ForestClassifier
        model = ForestClassifier(random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def logistic_regression(self, target=None, col=[], model_type = "Logistic Regression"):
        """This function fits a logistic regression model to the training data"""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter = 500, random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def xgboost_classifier(self, target=None, col=[], model_type = "XGBoost Classifier"):
        """This function fits an XGBoost classifier to the training data"""
        from xgboost import XGBClassifier
        model = XGBClassifier( eval_metric='logloss', random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def decision_tree_classifier(self, target=None, col=[], model_type = "Decision Tree Classifier"):
        """This function fits a decision tree classifier to the training data"""
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def sas_decision_tree_classifier(self, target=None, col=[], model_type = "SAS Decision Tree Classifier"):
        """This function fits a decision tree classifier to the training data"""
        from sasviya.ml.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def sas_gradboost_classifier(self, target=None, col=[], model_type = "SAS Gradient Boosting Classifier"):
        """This function fits an XGBoost classifier to the training data"""
        from sasviya.ml.tree import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def sas_forest_classifier(self, target=None, col=[], model_type = "SAS Forest Classifier"):
        """This function fits a SAS Forest classifier to the training data"""
        from sasviya.ml.tree import ForestClassifier
        model = ForestClassifier(random_state=42)
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def sas_support_vector_classifier(self, target=None, col=[], model_type = "SAS Support Vector Classifier"):
        """This function fits a SAS Support Vector Machine-based classifier to the training data"""
        from sasviya.ml.svm import SVC
        model = SVC()
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def sas_logistic_regression(self, target=None, col=[], model_type = "SAS Logistic Regression"):
        """This function fits a SAS Support Vector Machine-based classifier to the training data"""
        from sasviya.ml.linear_model import LogisticRegression
        model = LogisticRegression()
        model_package = self.run_model(model, target, col)
        model_package ["model_type"] = model_type
        self.models_run.append(model_package)

    def save(self, path=None):
        """This function saves the PharmMod class object to the path provided"""
        import pickle
        import os
        path_check = self.path_check(path)
        if path_check["extn"] == ".pkl":
            if path_check["exists"] == False:
                with open(path, 'wb') as f:
                    pickle.dump(self, f)

    def load(self, path=None):
        """This function loads the PharmMod class object from the path provided"""
        import pickle
        path_check = self.path_check(path)
        if path_check["extn"] == ".pkl":
            with open(path, 'rb') as f:
                self = pickle.load(f)
            return self
        else:
            print("Unsupported file format")
            return None        
        
