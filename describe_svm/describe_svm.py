from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import random

# setting the random seed that will be used anywhere a random state needs to be set
np.random.seed(444924)
random.seed(444924)
seed = 444924

class svm_description:
    def __init__(self, features, labels):
        # need to use this in the desc_feature_rel method
        self.features_raw = features

        # calculating the mutual information (MI) between each feature and the label variable in order to determine...
        # which features add mor512e "information" to the model
        self.features_mi = mutual_info_classif(features, labels,
                                              discrete_features=[True, True, False, True, True, False, False, False, False],
                                              random_state=seed)
        self.features_mi = pd.DataFrame(data=self.features_mi, index=features.columns, columns=["Mutual Information"])

        # there are 4 sets of features
        # features in the original feature space
        self.features = StandardScaler().fit_transform(features)
        # features reduced to 2d using PCA
        self.features_2d = PCA(n_components=2).fit_transform(self.features)
        # features excluding those with low MI (decided to use 0.01 as threshold)
        informative = list(self.features_mi[self.features_mi["Mutual Information"] >= 0.01].index)
        # used for testing
        # informative = ["Age", "Educ", "SES", "MMSE", "eTIV", "ASF"]
        self.features_filtered = features[informative].copy()
        self.features_filtered = StandardScaler().fit_transform(self.features_filtered)
        # features excluding those with low MI reduced to 2d using PCA
        self.features_filtered_2d = PCA(n_components=2).fit_transform(self.features_filtered)

        self.labels = labels

        # there are 4 sets of training and testing data
        # train/test using all features, original dimensions
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.features, 
                                                                            self.labels, 
                                                                            test_size= 0.3, 
                                                                            random_state=seed)
        # train/test using all features, 2d
        self.xtrain_2d, self.xtest_2d, self.ytrain_2d, self.ytest_2d = train_test_split(self.features_2d, 
                                                                                        self.labels, 
                                                                                        test_size= 0.3, 
                                                                                        random_state=seed)
        # train/test excluding features with low MI, original dimensions
        self.xtrain_flt, self.xtest_flt, self.ytrain_flt, self.ytest_flt = train_test_split(self.features_filtered, 
                                                                                            self.labels, 
                                                                                            test_size= 0.3, 
                                                                                            random_state=seed)
        # train/test excluding features with low MI, 2d
        self.xtrain_flt_2d, self.xtest_flt_2d, self.ytrain_flt_2d, self.ytest_flt_2d = train_test_split(self.features_filtered_2d, 
                                                                                                        self.labels, 
                                                                                                        test_size= 0.3, 
                                                                                                        random_state=seed)

        # if the "median trick" is used to supply the gamma parameter, it is calculated as below...
        # calculating a gamma parameter value for the SVM/SVC model using a rbf kernel using the "median trick"
        # randomly choose some samples to use in the calculation
        samples = np.random.choice(self.xtrain_flt.shape[0], size=100, replace=False)
        # split the sample population in half
        x_i, x_j = samples[:50], samples[50:]
        x_ij = np.hstack((self.xtrain_flt[x_i, :], self.xtrain_flt[x_j, :]))
        # find the L2 norm distance between pairs of points from the 2 sample sets and find the median of the distances
        M = np.median(np.linalg.norm(x_ij, ord=2, axis=1))
        sigma = M**(1/2)
        trick_gamma = 1/(2*(sigma**2))

        # it is time consuming to run k-fold cross-validation each time, therefore commenting this section out
        # parameters = {"kernel": ["rbf"], "probability": [True], "class_weight": ["balanced"], 
        #               "C": np.arange(1, 11, 1), "gamma": ["scale", "auto", trick_gamma]}
        # cv_svm = GridSearchCV(estimator=SVC(), param_grid=parameters, scoring="roc_auc", cv=5)
        # cv_svm.fit(self.xtrain_flt, self.ytrain_flt)
        # print(cv_svm.best_estimator_)
        # print(cv_svm.best_params_)
        # print(cv_svm.best_score_)

        # there are 4 models, one for each of the feature sets
        self.model = SVC(kernel="rbf", probability=True, 
                         class_weight="balanced", C=2, 
                         gamma="scale", random_state=seed).fit(self.xtrain, self.ytrain)
        self.model_2d = SVC(kernel="rbf", probability=True, 
                            class_weight="balanced", C=2,
                            gamma="scale", random_state=seed).fit(self.xtrain_2d, self.ytrain_2d)
        self.model_filtered = SVC(kernel="rbf", probability=True,
                                  class_weight="balanced", C=1, 
                                  gamma=trick_gamma, random_state=seed).fit(self.xtrain_flt, self.ytrain_flt)
        self.model_filtered_2d = SVC(kernel="rbf", probability=True,
                                     class_weight="balanced", C=1,
                                     gamma=trick_gamma, random_state=seed).fit(self.xtrain_flt_2d, self.ytrain_flt_2d)

        # 2 accuracy scores, one for each of the models
        self.accuracy = self.model.score(self.xtest, self.ytest)
        self.accuracy_filtered = self.model_filtered.score(self.xtest_flt, self.ytest_flt)

    def plot_overview(self, version):
        assert version in ["all", "filtered"], "version should be 'all' or 'filtered'"
        if version == "all":
            ftr = self.features_2d
        elif version == "filtered":
            ftr = self.features_filtered_2d
        colors = ["darkblue", "darkgreen"]
        plt.figure()
        self.__plot_decision_boundary(version)
        # initially, there was going to be more than 2 classes, therefore a for loop to plot each of...
        # the classes made more sense, but due to lack of examples of more extreme cases...
        # there will be only 2 classes but keeping the for loop since it is already complete
        for i in range(len(np.unique(self.labels))):
            label = np.unique(self.labels)[i]
            # choose a new color from the list for each class
            color = colors[i]
            class_x = np.where(self.labels == label)[0]
            # plotting must be done using the 2d representation
            plt.scatter(ftr[class_x, 0], ftr[class_x, 1], c=color, marker="x", label=f"class_{label}")
        # might need to fix labels, 0 should be no symptoms of dementia, 1 should be symptoms of dementia
        plt.legend(loc="best")
        plt.savefig(f"overview_{version}.png")
        plt.close()

    def __plot_decision_boundary(self, version):
        assert version in ["all", "filtered"], "version should be 'all' or 'filtered'"
        if version == "all":
            ftr = self.features_2d
            clf = self.model_2d
        elif version == "filtered":
            ftr = self.features_filtered_2d
            clf = self.model_filtered_2d
        # here a 'ListedColormap' is necessary because the countour is not being plotted in a for loop
        colors = ListedColormap(["lightblue", "lightgreen"])
        # y here does not represent labels, instead they represent the y axis of the plot
        # the features have been reduced to 2d so the x axis represents one of the reduced features...
        # and the y axis represents the other reduced feature
        # must use the 2d representation for plotting
        xmin, xmax = ftr[:, 0].min() - 1, ftr[:, 0].max() + 1
        ymin, ymax = ftr[:, 1].min() - 1, ftr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))
        Z = clf.predict(np.column_stack((xx.ravel(), yy.ravel())))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=colors)

    def plot_support_vectors(self, version):
        assert version in ["all", "filtered"], "version should be 'all' or 'filtered'"
        if version == "all":
            ftr = self.features_2d
            clf = self.model_2d
        elif version == "filtered":
            ftr = self.features_filtered_2d
            clf = self.model_filtered_2d
        colors = ["lightblue", "lightgreen"]
        support_vectors = clf.support_vectors_
        plt.figure()
        # initially, there was going to be more than 2 classes, therefore a for loop to plot each of...
        # the classes made more sense, but due to lack of examples of more extreme cases...
        # there will be only 2 classes but keeping the for loop since it is already complete
        for i in range(len(np.unique(self.labels))):
            label = np.unique(self.labels)[i]
            color = colors[i]
            class_x = np.where(self.labels == label)[0]
            plt.scatter(ftr[class_x, 0], ftr[class_x, 1], c=color, marker="x", label=f"class_{label}")
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c="darkred", marker="+", label="support vectors")
        plt.legend(loc="best")
        plt.savefig(f"support_vectors_{version}.png")
        plt.close()

    def calc_sv_ratios(self, version):
        assert version in ["all", "filtered"], "version should be 'all' or 'filtered'"
        if version == "all":
            print("all")
            labels = self.ytrain
            clf = self.model
        elif version == "filtered":
            print("filtered")
            labels = self.ytrain_flt
            clf = self.model_filtered
        ttl_dat = len(labels)
        ttl_sv = len(clf.support_vectors_)
        print(f"total # of training datapoints = {ttl_dat}")
        print(f"total # of sv = {ttl_sv}")
        print(f"total # of sv / total # of training datapoints = {ttl_sv / ttl_dat}")
        for i in range(len(clf.classes_)):
            label = clf.classes_[i]
            class_dat = len(np.where(labels == label)[0])
            class_sv = clf.n_support_[i]
            print(f"# of training datapoints from class {label} = {class_dat}")
            print(f"# of sv from class {label} = {class_sv}")
            print(f"# of sv from class {label} / # of training datapoints from class {label} = {class_sv / class_dat}")

    def desc_feature_rel(self):
        # I know that this is a lot of nested loops
        # number of neighbors (support vectors)
        k = 1
        # number of samples from each class
        s = 30
        support_vector_ind = self.model.support_
        # train a knn model on the support vectors, use this to find the support vectors nearest a non-support vector
        knn = NearestNeighbors(n_neighbors=k).fit(self.model.support_vectors_)
        # for each class, randomly select s non-support vectors
        for i in range(len(np.unique(self.labels))):
            label = np.unique(self.labels)[i]
            class_x_ind = np.where(self.labels == label)[0]
            class_x_samples = []
            while len(class_x_samples) < s:
                sample_ind = random.choice(class_x_ind)
                if (sample_ind not in support_vector_ind) and (sample_ind not in class_x_samples):
                    class_x_samples.append(sample_ind)
            sample_deltas = []
            for sample in class_x_samples:
                new_point = self.features[sample].reshape(-1, self.model.support_vectors_.shape[1])
                # for each of the randomly selected non-support vectors, find the k nearest support vectors...
                # and the distances to those k neighbors
                # n_neighbors=k being set in knn constructor above
                dist, nearest_ind = knn.kneighbors(new_point)
                nearest = self.features[nearest_ind]
                deltas = []
                # for each feature of the sample datapoint...
                # individually (keeping all other features the same) test the difference a p% increase in the feature makes...
                # to the sample datapoint's distance to each of its nearest neighbors
                p = 1
                for fi in range(len(new_point[0])):
                    f = new_point[0][fi]
                    f_plus = f + (f * p)
                    new_point_plus = new_point.copy()
                    new_point_plus[0][fi] = f_plus
                    # calculate the L2 norms to each of the neighbors after the change
                    dist_plus = [np.linalg.norm(new_point_plus-n) for n in nearest[0]]
                    # find the absolute value change in distance from each of the nearest support vectors
                    # take the abolute value to have to avoid testing increases and decreases separately
                    deltas.append(np.absolute(dist[0] - dist_plus))
                deltas = np.sum(np.array(deltas), axis=1)
                sample_deltas.append(deltas)
            sample_deltas = np.mean(np.array(sample_deltas), axis=0)
            sample_deltas = pd.DataFrame(data=sample_deltas, 
                                         index=self.features_raw.columns, 
                                         columns=["Avg change in distance"])
            print(f"class_{label}")
            print(sample_deltas)
            print("\n")



                