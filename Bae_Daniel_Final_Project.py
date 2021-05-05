import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# importing local 'describe_svm' package which contains the 'svm_description' class
from describe_svm.describe_svm import svm_description

def main():
    # all data is in the data folder within the project directory
    data_path = os.path.join("data", "mri_data.csv")
    mri_data = pd.read_csv(data_path)

    # the data contains one binary features, 'M/F', which indicates whether a patient is male or female
    # using one hot encoding to transform this features into two columns, 'Male' and 'Female'
    # e.g. if a patient is male, the column 'Male' will have a value 1, otherwise 0
    features = mri_data[["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF"]].copy()
    gender = np.array(mri_data["M/F"]).reshape(-1, 1)
    gender_bin = OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(gender)
    features.insert(loc=0, column="Female", value=gender_bin[:, 0], allow_duplicates=False)
    features.insert(loc=0, column="Male", value=gender_bin[:, 1], allow_duplicates=False)

    # initially the intention was to have 3 or 4 classes but there were not enough examples...
    # of patients with more severe dementia
    # if class is 0, patient exhibits no dementia
    # if class is 1, patient exhibits some symptoms of dementia, whether minor or severe
    labels = mri_data["CDR"].copy()
    # if x != 0, True is returned, int(True) == 1
    labels = [int(x != 0) for x in labels]
    labels = LabelEncoder().fit_transform(labels)
    
    # creating a svm_description class instance that will track and describe results of analyses
    boundaries = svm_description(features, labels)

    # all plotting is done using the data that has been reduced to 2d
    # # run this to plot all of the data with the decision boundary developed on the training data
    # boundaries.plot_overview("filtered")
    # # run this to plot all of the data with the support vectors
    # boundaries.plot_support_vectors("filtered")
    # # run this to output the support vector to data ratios
    # boundaries.calc_sv_ratios("filtered")
    # # run this to describe features based on their influence on...
    # # datapoint distance from support vectors
    # boundaries.desc_feature_rel()

if __name__ == "__main__":
    main()