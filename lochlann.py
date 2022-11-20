# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:11:20 2022

@author: lochl
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings


warnings.filterwarnings('ignore')
#SAMPLE_SIZE = 600
SAMPLE_SIZES = [800, 1600, 3200, 6400, 12800]
KFOLD_SIZE = 10


def avg(list):
    return sum(list) / len(list)


def plot_image(image):
    
    plt.imshow(np.array(image).reshape(28, 28))
    plt.show()

def preprocess_data(filename):
    
    df_images = pd.read_csv(filename)
    features = df_images.drop("label", axis=1)
    labels = df_images["label"]
    return features, labels

def parameterize_data(param_size, features, labels):
    
    features_param = features.sample(param_size)
    labels_param = labels[features_param.index]
    return features_param, labels_param

def evaluate_classifier(size_kfolds, classifier, sample_size, features, labels):
    
    print("\n\n------------------------------------------------")
    print("\n------------", classifier)
    print("\n------------ Sample Size:\t", sample_size)
    print("\n------------------------------------------------")
    
    summary = {
        "training_times":[],
        "prediction_times":[],
        "prediction_accuracies":[]
    }
    
    kf = KFold()
    current_kfold = 0

    for index_train, index_test, in kf.split(features):
        
        features_train = features.iloc[index_train]
        features_test = features.iloc[index_test]
        labels_train = labels.iloc[index_train]
        labels_test = labels.iloc[index_test]
        
        print("{:<24}:\t{:<}".format("kfold index", current_kfold))
        
        time_start = time.time()
        training = classifier.fit(features_train, labels_train)
        time_finish = time.time()
        process_time = (time_finish - time_start) * 1000
        summary['training_times'].append(process_time)
        print("{:<24}:\t{:<.2f} ms".format("Training Time", process_time))
        
        time_start = time.time()
        prediction = classifier.predict(features_test)
        time_finish = time.time()
        process_time = (time_finish - time_start) * 1000
        summary['prediction_times'].append(process_time)
        print("{:<24}:\t{:<.2f} ms".format("Testing Time", process_time))
        
        prediction_accuracy = accuracy_score(labels_test, prediction)
        summary['prediction_accuracies'].append(prediction_accuracy)
        print("{:<24}:\t{:<.2f} %".format("Prediction Accuracy", (prediction_accuracy)*100))
        
        confusion = confusion_matrix(labels_test, prediction)
        print("{:<24}:\t".format("Confusion Matrix"))
        print(confusion)
        #tb = confusion.tobytes()
        #fb = (np.frombuffer(ts, dtype=int))
        #print("{:<32}:\t{}".format("Confusion Matrix", fs)) #FIXME - how to remove whitespace formatting for confusion_matrix print????

        print()
        current_kfold += 1

    #print("-------- SUMMARY OF", classifier, "--------")
    print("{:<32}:\t{:<.2f} ms".format("Training Time (min)", min(summary['training_times'])))
    print("{:<32}:\t{:<.2f} ms".format("Training Time (max)", max(summary['training_times'])))
    print("{:<32}:\t{:<.2f} ms".format("Training Time (avg)", avg(summary['training_times'])))
    print("{:<32}:\t{:<.2f} ms".format("Prediction Time (min)", min(summary['prediction_times'])))
    print("{:<32}:\t{:<.2f} ms".format("Prediction Time (max)", max(summary['prediction_times'])))
    print("{:<32}:\t{:<.2f} ms".format("Prediction Time (avg)", avg(summary['prediction_times'])))
    print("{:<32}:\t{:<.2f} %".format("Prediction Accuracy (min)", (min(summary['prediction_accuracies']))*100))
    print("{:<32}:\t{:<.2f} %".format("Prediction Accuracy (max)", (max(summary['prediction_accuracies']))*100))
    print("{:<32}:\t{:<.2f} %".format("Prediction Accuracy (avg)", (avg(summary['prediction_accuracies']))*100))
   
    return summary
        
def evaluate_perceptron(sample_size, features, labels):
    
    classifier = linear_model.Perceptron()
    summary = evaluate_classifier(KFOLD_SIZE, classifier, sample_size, features, labels)
    return summary

def evaluate_SVM(sample_size, features, labels):
    
    #use a radial basis function kernel
    #gammas = [1e-1, 1e-3, 1e-5, 1e-7]
    #for gamma in gammas:
    #    classifier = svm.SVC(kernel="rbf", gamma=gamma)
    #    summary = evaluate_classifier(KFOLD_SIZE, classifier, features, labels)
    
    classifier = svm.SVC(kernel="rbf", gamma=1e-7)
    summary = evaluate_classifier(KFOLD_SIZE, classifier, sample_size, features, labels)
    return summary # return summary of last evaluation (1e-7), prediction accuracy > 90%

def evaluate_knearest(sample_size, features, labels): # FIXME

    #neighbours = [1, 3, 5, 7, 9, 11]
    #for neighbour in neighbours:
    #    classifier = KNeighborsClassifier(n_neighbors=neighbour)
    #    summary = evaluate_classifier(KFOLD_SIZE, classifier, features, labels)
    
    classifier = KNeighborsClassifier(n_neighbors=3) # FIXME - why is there an error message in the terminal, despite working fine?????
    summary = evaluate_classifier(KFOLD_SIZE, classifier, sample_size, features, labels)
    return summary

def evaluate_dtree(sample_size, features, labels):
    classifier = DecisionTreeClassifier()
    summary = evaluate_classifier(KFOLD_SIZE, classifier, sample_size, features, labels)
    return summary



def main():
    #------------------------------------------------------------------------------------
    #----------------- TASK 1 (pre-processing and visualisation, 5 points ----------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -   Load the product dataset and separate the labels [1point]
    # COMPLETED    -   from the feature vectors
    # COMPLETED    -   how many samples are images of sneakers, how many sameples are images of ankle boots [1point]
    # COMPLETED    -   display at least one for each class [2point]
    print("\n\n------------------------------------------------")
    print("\n--------------- Pre-process data")
    print("\n------------------------------------------------")
    features, labels = preprocess_data('product_images.csv')
    
    print("{:<8s}:\t{:>}".format("Data", "FEATURE VECTORS"))
    print("{:<8s}:\t".format("Type"), end="")
    print(type(features))
    print(features)
    
    print("\n{:<8s}:\t{:>}".format("Data", "LABELS"))
    print("{:<8s}:\t".format("Type"), end="")
    print(type(labels))
    print(labels)
    
    #Show info on SNKEAKER accompanied with example plot
    plot_image(features.iloc[labels[labels == 0].index[0]])
    print("\n{:<8s}:\t{:>}".format("Shoe", "SNEAKER"))
    print("{:<8s}:\t{:>}".format("Label", 0))
    print("{:<8s}:\t{:>}".format("Count", labels[labels == 0].size))
    
    #Show info on ANKLE BOOTS accompanied with example plot
    plot_image(features.iloc[labels[labels == 1].index[0]])
    print("\n{:<8s}:\t{:>}".format("Shoe", "ANKLE BOOT"))
    print("{:<8s}:\t{:>}".format("Label", 1))
    print("{:<8s}:\t{:>}".format("Count", labels[labels == 1].size))

    #------------------------------------------------------------------------------------
    #---------------------- TASK 2 (evaluation procedure, 9 points) ---------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -   Create a k-fold cross-validation procedure to split the data into training [1point]
    # COMPLETED    -   and evaluation subset [1point]
    # COMPLETED    -   parameterize the number of samples to use from the dataset to be able to control the runtime of the algorithm evaluation [1point]
    # COMPLETED    -   start developing using a small number of samples and increase for the final evaluation
    # COMPLETED    -   measure for each split of the cross-validation procedure the processing time required for training [1point]
    # COMPLETED    -   the processing time required for prediction [1point]
    # COMPLETED    -   determine the confusion matrix [1point]
    # COMPLETED    -   and accuracy score of the classification
    # COMPLETED    -   calulate the minimum, maximum, and average of :
    # COMPLETED    -         the training time per training sample [1point]
    # COMPLETED    -         the prediction time per evaluation sample [1point]
    # COMPLETED    -         the prediction accuracy [1point]
    list_summary_perceptron = []
    list_summary_SVM = []
    list_summary_knearest = []
    list_summary_dtree = []
    
    for sample_size in SAMPLE_SIZES:
        features_param, labels_param = parameterize_data(sample_size, features, labels)
        
        print("\n\n--------------------------------------------------")
        print("\n----------- New Parameterized Sample Size")
        print("\n----------- Sample Size:\t", sample_size)
        print("\n--------------------------------------------------")
        print("{:<32s}:\t{:>}".format("PARAMETERIZED DATA SAMPLE SIZE", sample_size))
        print("{:<32s}:\t{:>}".format("Count Feature Vectors (param)", features_param.size))
        print("{:<32s}:\t{:>}".format("Count Labels (param)", labels_param.size))
        print("{:<32s}:\t{:>}".format("Count Sneakers (param)", labels_param[labels_param == 0].size))
        print("{:<32s}:\t{:>}".format("Count Ankle Boots (param)", labels_param[labels_param == 1].size))
        
        list_summary_perceptron.append(evaluate_perceptron(sample_size, features_param, labels_param))
        list_summary_SVM.append(evaluate_SVM(sample_size, features_param, labels_param))
        list_summary_knearest.append(evaluate_knearest(sample_size, features_param, labels_param))
        list_summary_dtree.append(evaluate_dtree(sample_size, features_param, labels_param))
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 3 (Perceptron, 3 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -  Use the procedure developed in task 2 to train and evaluate the Perceptron classifier [1 point]
    # COMPLETED    -  What is the mean prediction accuracy of this classifier [1 point]
    # COMPLETED    -  Vary the number of samples and plot the relationship between input data size and runtimes for the classifier [1 point].
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 4 (Support Vector Machine, 5 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -  Use the procedure developed in task 2 to train and evaluate the Support Vector Machine classifier [1 point]
    # COMPLETED    -  Use a radial basis function kernel and try different choices for the parameter 𝛾 [1 point]
    # COMPLETED    -  Determine a good value for 𝛾 based on mean prediction accuracy [1 point]
    # COMPLETED    -  What is the best achievable mean prediction accuracy of this classifier [1 point]
    # COMPLETED    -  Determine a good value for 𝛾 based on mean prediction accuracy [1 point]
    # COMPLETED    -  Vary the number of samples and plot the relationship between input data size and runtimes for the optimal classifier [1 point]
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 5 (k-nearest Neighbours, 5 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -  Use the procedure developed in task 2 to train and evaluate the k-nearest neighbour classifier [1 point]
    # COMPLETED    -  Try different choices for the parameter k [1 point]
    # COMPLETED    -  and determine a good value based on mean prediction accuracy [1 point]
    # COMPLETED    -  What is the best achievable mean prediction accuracy of this classifier [1 point]
    # COMPLETED    -  Vary the number of samples and plot the relationship between input data size and runtimes for the optimal classifier [1 point]
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 6 (Decision trees, 3 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -  Vary the number of samples and plot the relationship between input data size and runtimes for the optimal classifier [1 point]
    # COMPLETED    -  What is the mean prediction accuracy of this classifier [1 point]
    # COMPLETED    -  Vary the number of samples and plot the relationship between input data size and runtimes for the classifier [1 point]
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 7 (comparison, 5 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED    -  Compare the training and prediction times of the four classifiers. What trend do you observe for each of the classifiers and why [4 points]
    # TODO    -  Also taking the accuracy into consideration, how would you rank the four classifiers and why [1 point]
    print("\n\n--------------------------------------------------")
    print("\n------------------- Comparison")
    print("\n--------------------------------------------------")
    
    
    # Comparing Classifier Prediction Accuracies
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_accuracies'])for s in list_summary_perceptron)), label='Perceptron', color='red')
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_accuracies'])for s in list_summary_SVM)), label='Support Vector Machine (gamma=1e-07)', color='green')
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_accuracies'])for s in list_summary_knearest)), label='K-Nearest Neighbour (n_neighbors=3)', color='blue')
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_accuracies'])for s in list_summary_dtree)), label='Decision Tree', color='orange')
    plt.title("Comparing Classifier Prediction Accuracies")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Accuracy (%)")
    plt.legend()
    plt.show()
    
    
    
    
    # Comparing Classifier Processing Times
    figure(figsize=(10,12),dpi=80)
    
    plt.subplot(2, 2, 1)
    plt.plot(SAMPLE_SIZES, (list(avg(s['training_times'])for s in list_summary_perceptron)), label="Training")
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_times'])for s in list_summary_perceptron)), label="Prediction")
    plt.title("Perceptron")
    plt.xlabel("Sample Size")
    plt.ylabel("Processing time (ms)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(SAMPLE_SIZES, (list(avg(s['training_times'])for s in list_summary_SVM)), label="Training")
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_times'])for s in list_summary_SVM)), label="Prediction")
    plt.title("Support Vector Machine (gamma=1e-07)")
    plt.xlabel("Sample Size")
    plt.ylabel("Processing time (ms)")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(SAMPLE_SIZES, (list(avg(s['training_times'])for s in list_summary_knearest)), label="Training")
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_times'])for s in list_summary_knearest)), label="Prediction")
    plt.title("K-Nearest Neighbour (n_neighbors=3)")
    plt.xlabel("Sample Size")
    plt.ylabel("Processing time (ms)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(SAMPLE_SIZES, (list(avg(s['training_times'])for s in list_summary_dtree)), label="Training")
    plt.plot(SAMPLE_SIZES, (list(avg(s['prediction_times'])for s in list_summary_dtree)), label="Prediction")
    plt.title("Decision Tree")
    plt.xlabel("Sample Size")
    plt.ylabel("Processing time (ms)")
    plt.legend()
    
    plt.suptitle("Comparing Classifier Processing Times")
    plt.show()
    

main()