import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# App ki heading
st.write(''' # Explore different ML models and datasets
Dekhtay han kon sa best ha inmay say?''')

# Dataset ka nam ik box may dal ker sidebar pay laga do
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine'))

# Iris k nichay classifier k nam ik box may dal do
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

# Dataset ko load kernay k liya ak function define kerna ha 
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y 

# Ab is function ko bula ker X, y variable ka equal rakh layna ha
X, y = get_dataset(dataset_name)

# Ab apnay dataset ki shape ko app per print ker dayn gay
st.write('Shape of Dataset:', X.shape)
st.write('Number of Classes', len(np.unique(y)))

# Next hum different classifier k parameter ko user input may add kerain gay
def add_parameter_ui(classifier_name):
    params = dict()                           # Create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C                       # Its the degree of correct  classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K                       # Its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth       # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # Number of trees
    return params

# Ab is function ko bula lay gay or params variable k equal rakh layn gay
params = add_parameter_ui(classifier_name)

# Ab hum classifier banayen gay on the bases of classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

# Ab is function ko call ker ka clf variable k equal rakh lyna ha 
clf = get_classifier(classifier_name, params) 

# Ab dataset ko test or train data may split ker laytay han by 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) 

# Ab hum nay apnay classifier ki training kerni ha 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Model ka accuracy score check ker lyna ha or isay App pay print ker dayna ha 
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc) 

#### PLOT DATASET ####
# Ab hum apnay saray features ko 2 dimensional plot pay draw ker dayn gay using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Ab hum apna data 0 or 1 dimension may slice ker day gay
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, 
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)