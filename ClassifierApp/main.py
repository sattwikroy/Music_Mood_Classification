from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn import preprocessing, svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

url = "F:\Minor Project\Training_data_Set.csv"

dataset = pd.read_csv(url) 

moods = list(set(dataset["mood"]))

# Encodig Y Labels
y_labelecoder = preprocessing.LabelEncoder()
dataset["mood"] = y_labelecoder.fit_transform(list(dataset["mood"]))

# Encodig Non Numeric Data
# le = preprocessing.LabelEncoder()
# le.fit(list(dataset["camelot"]))
# dataset["camelot"] = le.transform(list(dataset["camelot"]))

# Setting X & Y from Numeric data
X = dataset.iloc[:, 2:11].values
y = dataset.iloc[:, 11].values 


# Spliting Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10) 


#Setting up Classification Models

acc = []
classifiers = {}

#========================== KNeighbours Classifier =========================

classifiers["KNN"]=KNeighborsClassifier(n_neighbors=5)
classifiers["KNN"].fit(X_train, y_train) 
acc.append(classifiers["KNN"].score(X_test, y_test))

print("========================== | KNN Prediction Report | =========================")

print("Accuracy : ", acc[0])

y_predict = classifiers["KNN"].predict(X_test)

print(classification_report(y_test, y_predict)) 

#---------------------- CFM of KNN -------------------------

cm = confusion_matrix(y_test, y_predict)
ax = sns.heatmap(cm, annot=True, cmap='Greens')
ax.set_title('Confusion Matrix for KNeighbours Classifier\n\n')
plt.savefig('KNN CFM',dpi=300)
plt.clf()

#....................... ROC of KNN ........................

y_prob_pred = classifiers["KNN"].predict_proba(X_test)

fpr = {}
tpr = {}
thresh ={}

colors = [ 'pink', 'darkred', 'seagreen', 'orange', 'teal', 'darksalmon', 'darkgreen', 'skyblue', 'khaki', 'lightcoral', 'black']

for i in range(len(moods)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob_pred[:,i], pos_label=i)
    
# plotting
for i in range(len(moods)):  
    plt.plot(fpr[i], tpr[i], linestyle='--',color=colors[i], label=moods[i]+' vs Rest')

plt.title('ROC curve of KNeighbours Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('KNN ROC',dpi=300)
plt.clf()

#========================== Gaussian Naive Bayes ===========================

classifiers["GNB"]=GaussianNB()
classifiers["GNB"].fit(X_train, y_train)
acc.append(classifiers["GNB"].score(X_test, y_test))

print("========================== | GNB Prediction Report | ===========================")

print("Accuracy : ", acc[1])

y_predict = classifiers["GNB"].predict(X_test)

print(classification_report(y_test, y_predict)) 

#---------------------- CFM of GNB -------------------------

cm = confusion_matrix(y_test, y_predict)
ax = sns.heatmap(cm, annot=True, cmap='Reds')
ax.set_title('Confusion Matrix for Gaussian Naive Bayes\n\n')
plt.savefig('GNB CFM',dpi=300)
plt.clf()

#....................... ROC of GNB ........................

y_prob_pred = classifiers["GNB"].predict_proba(X_test)

fpr = {}
tpr = {}
thresh ={}

for i in range(len(moods)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob_pred[:,i], pos_label=i)
    
# plotting
for i in range(len(moods)):  
    plt.plot(fpr[i], tpr[i], linestyle='--',color=colors[i], label=moods[i]+' vs Rest')

plt.title('ROC Curve of Gaussian Naive Baye')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('GNB ROC',dpi=300)
plt.clf()


#========================== Support Vector Machine ==========================

classifiers["SVM"]=svm.SVC(kernel="rbf", C=1000, gamma=0.001, probability=True)
classifiers["SVM"].fit(X_train, y_train)
acc.append(classifiers["SVM"].score(X_test,y_test))


print("========================== | SVM  Prediction Report | ==========================")

print("Accuracy : ", acc[2])

y_predict = classifiers["SVM"].predict(X_test)

print(classification_report(y_test, y_predict))

#---------------------- CFM of SVM -------------------------

cm = confusion_matrix(y_test, y_predict)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix for Support Vector Machine\n\n')
plt.savefig('SVM CFM',dpi=300)
plt.clf()


#....................... ROC of SVM ........................

y_prob_pred = classifiers["SVM"].predict_proba(X_test)

fpr = {}
tpr = {}
thresh ={}

for i in range(len(moods)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob_pred[:,i], pos_label=i)
    
# plotting
for i in range(len(moods)):  
    plt.plot(fpr[i], tpr[i], linestyle='--',color=colors[i], label=moods[i]+' vs Rest')

plt.title('ROC curve of Support Vector Machine')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('SVM ROC',dpi=300)
plt.clf()

#Predict Function

def predict_mood(songdata):
    #songdata['camelot'] = le.transform([songdata['camelot']])
    values = list(songdata.values())
    ch = values.pop()
    prediction = classifiers[ch].predict([values])
    return y_labelecoder.inverse_transform(prediction)[0]
