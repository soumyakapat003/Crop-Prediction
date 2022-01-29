import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("CROP PREDICTION")

@st.cache

def readCsv(Csv):
  df = pd.read_csv(Csv)
  return df
Csv = 'Crop1.csv'
df = readCsv(Csv)
#st.write(df)

shape = df.shape

# Splitting the dataset into input and output values
def SplitInput(df,a,b):
  X = df.iloc[:, a:b]
  return X
def SplitOutput(df,a):
  y = df.iloc[:, a]
  return y
X = SplitInput(df,0,7)
y = SplitOutput(df,7)

st.write("### Explore dataset")

data = st.sidebar.selectbox("Selecting for Dataset exploring",
                     ("None","Whole Dataset","Number of columns","Number of rows","Null value checking","Input Data","Output Data","Unique Output values"))

if data == "None":
    st.write("")
elif data == "Whole Dataset":
    st.write("Whole Dataset",df)
elif data == "Null value checking":
    st.write("Null value checking",df.isnull().sum())
elif data == "Number of columns":
    st.write("Number of columns: ",shape[1])
elif data == "Number of rows":
    st.write("Number of rows: ",shape[0])
elif data == "Input Data":
    st.write("Input Data",X)
elif data == "Output Data":
    st.write("Output Data",y)
elif data == "Unique Output values":
    unique_y = np.unique(y,return_counts=True)
    V = np.array(unique_y).T
    Values = pd.DataFrame(V)
    st.write(Values)

st.write("### Visualization the dataset")

data_visualization = st.sidebar.selectbox("Select features vs Crop type for visualization",
                     ("None","Ph Vs Crop type","Humidity Vs Crop type","Temperature Vs Crop type","Rainfall Vs Crop type","Potassium Vs Crop type","Nitrogen Vs Crop type","Phosphorus Vs Crop type"))

def Graphs(data_visualization):
  if data_visualization == "None":
      st.write("")
  elif data_visualization == "Ph Vs Crop type":
      st.write("Ph Vs Crop type")
      sns.barplot(x='Croptype',y='ph',data=df)
      plt.xticks(rotation=270)
      plt.title("Ph Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Ph")
      st.pyplot()
  elif data_visualization == "Humidity Vs Crop type":
      st.write("Humidity Vs Crop type")
      sns.barplot(x='Croptype',y='humidity',data=df,palette='bright')
      plt.xticks(rotation=270)
      plt.title("Humidity Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Humidity")
      st.pyplot()
  elif data_visualization == "Temperature Vs Crop type":
      st.write("Temperature Vs Crop type")
      sns.barplot(x='Croptype',y='temperature',data=df,palette='dark')
      plt.xticks(rotation=270)
      plt.title("Temperature Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Temperature")
      st.pyplot()
  elif data_visualization == "Rainfall Vs Crop type":
      st.write("Rainfall Vs Crop type")
      sns.barplot(x='Croptype',y='rainfall',data=df,palette='colorblind')
      plt.xticks(rotation=270)
      plt.title("Rainfall Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Rainfall")
      st.pyplot()
  elif data_visualization == "Potassium Vs Crop type":
      st.write("Potassium Vs Crop type")
      K1 = X.iloc[:,2]
      plt.bar(y,K1,color="Red")
      plt.xticks(rotation=270)
      plt.title("Potassium Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Potassium")
      st.pyplot()
  elif data_visualization == "Nitrogen Vs Crop type":
      st.write("Nitrogen Vs Crop type")
      N1 = X.iloc[:,0]
      plt.bar(y,N1,color="Green")
      plt.xticks(rotation=270)
      plt.title("Nitrogen Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("Nitrogen")
      st.pyplot()
  elif data_visualization == "Phosphorus Vs Crop type":
      st.write("Phosphorus  Vs Crop type")
      P1 = X.iloc[:,1]
      plt.bar(y,P1,color="Magenta")
      plt.xticks(rotation=270)
      plt.title("phosphorus Vs Crop type")
      plt.xlabel("Crop type")
      plt.ylabel("phosphorus")
      st.pyplot()
Graphs(data_visualization)

st.write("### Explore different classification model")

classifier_name = st.sidebar.selectbox("Select Classifier Model",
                     ("None","Decision Tree","Random Forest","K Nearest Neighbor","Logistic Regression"))


# Encoding the data (Lebel Encoding)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y)

    

def parameter(clf_name):
    param = dict()
    if clf_name == "None":
        st.write("")
    elif clf_name == "Decision Tree":
        criterion = st.sidebar.selectbox("Select criterion",("gini","entropy"))
        param["criterion"] = criterion
        max_d= st.sidebar.slider("max_depth",2,15)
        param["max_d"] = max_d
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators",1,1000)
        param["n_estimators"] = n_estimators
        max_depth = st.sidebar.slider("max_depth",2,20)
        param["max_depth"] = max_depth
        crit = st.sidebar.selectbox("Select criterion",("gini","entropy"))
        param["crit"] = crit
    elif clf_name == "K Nearest Neighbor":
        n_neighbors = st.sidebar.slider("n_neighbors",1,50)
        param["n_neighbors"] = n_neighbors
        weights = st.sidebar.selectbox("Select weights",("uniform","distance"))
        param["weights"] = weights
    return param

param = parameter(classifier_name)

def classifier(clf_name,param):
    if clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(criterion = param["criterion"], max_depth = param["max_d"], random_state = 42)
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators = param["n_estimators"], max_depth = param["max_depth"], 
                                              criterion = param["crit"], random_state = 42)
    elif clf_name == "K Nearest Neighbor":
        clf = KNeighborsClassifier(n_neighbors = param["n_neighbors"], weights = param["weights"])
    else:
        clf = LogisticRegression(random_state = 42)
    return clf

clf = classifier(classifier_name,param)

# Spliting the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = .2, random_state = 42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)*100

st.write(f"Classifier = {classifier_name}")
if classifier_name == "None":
    st.write("Accuracy = None")
else:
    st.write(f"Accuracy = {accuracy}")
    
# Data Visualization: Actual Value Vs Predicted Value
if classifier_name == "None":
    st.write("")
else:
    st.write("Actual value vs Predicted value")
    f, ax = plt.subplots(figsize=(15,10))
    con_mat = confusion_matrix(y_test,y_pred)
    sns.heatmap(con_mat,annot=True,ax=ax)
    plt.title("Actual Value Vs Predicted Value")
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    st.pyplot()

# Prediction of Crop by user input
st.write("### Prediction by user input")
if classifier_name == "None":
    st.subheader("Predicted crop is: None")
else:
    a=st.number_input('Enter the Nitrogen value in soil: ')
    b=st.number_input('Enter the Phosphorus value in soil: ')
    c=st.number_input('Enter the Potassium value in soil: ')
    d=st.number_input('Enter the Temperature: ')
    e=st.number_input('Enter the Humidity: ')
    f=st.number_input('Enter the ph level of the soil: ')
    g=st.number_input('Enter the rainfall: ')
        
    if (a == 0 and b == 0 and c == 0 and d == 0 and e == 0 and f == 0 and g == 0):
        st.subheader("Predicted crop is: None")
    else:
        P = int(clf.predict([[a,b,c,d,e,f,g]]))
        #print(P)
        if P==0:
            st.subheader("Predicted crop is: apple")
        elif P==1:
            st.subheader("Predicted crop is: banana")
        elif P==2:
            st.subheader("Predicted crop is: blackgram")
        elif P==3:
            st.subheader("Predicted crop is: chickpea")
        elif P==4:
            st.subheader("Predicted crop is: coconut")
        elif P==5:
            st.subheader("Predicted crop is: coffee")
        elif P==6:
            st.subheader("Predicted crop is: cotton")
        elif P==7:
            st.subheader("Predicted crop is: grapes")
        elif P==8:
            st.subheader("Predicted crop is: jute")
        elif P==9:
            st.subheader("Predicted crop is: kidneybeans")
        elif P==10:
            st.subheader("Predicted crop is: lentil")
        elif P==11:
            st.subheader("Predicted crop is: maize")
        elif P==12:
            st.subheader("Predicted crop is: mango")
        elif P==13:
            st.subheader("Predicted crop is: mothbeans")
        elif P==14:
            st.subheader("Predicted crop is: mungbean")
        elif P==15:
            st.subheader("Predicted crop is: muskmelon")
        elif P==16:
            st.subheader("Predicted crop is: orange")
        elif P==17:
            st.subheader("Predicted crop is: papaya")
        elif P==18:
            st.subheader("Predicted crop is: pigeonpeas")
        elif P==19:
            st.subheader("Predicted crop is: pomegranate")
        elif P==20:
            st.subheader("Predicted crop is: rice")
        elif P==21:
            st.subheader("Predicted crop is: watermelon")
