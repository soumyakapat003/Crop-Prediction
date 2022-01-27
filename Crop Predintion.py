import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('E:\lenovo\SOUMYA\ML Project\CSV Folder\Crop1.csv')

# Exploring the Dataset
dataset.info()

# Null value checking
dataset.isnull().sum()

# Duplicate values checking
dataset.duplicated().sum()

# Splitting the dataset into input and output values
X = dataset.iloc[:, 0:7]
y = dataset.iloc[:, 7]

#Visualization the dataset for Ph Vs Crop type
sns.barplot(x='Croptype',y='ph',data=dataset)
plt.xticks(rotation=270)
plt.title("Ph Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Ph")
plt.show()

#Visualization the dataset for Humidity Vs Crop type
sns.barplot(x='Croptype',y='humidity',data=dataset,palette='bright')
plt.xticks(rotation=270)
plt.title("Humidity Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Humidity")
plt.show()

#Visualization the dataset for Temperature Vs Crop type
sns.barplot(x='Croptype',y='temperature',data=dataset,palette='dark')
plt.xticks(rotation=270)
plt.title("Temperature Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Temperature")
plt.show()

#Visualization the dataset for Rainfall Vs Crop type
sns.barplot(x='Croptype',y='rainfall',data=dataset,palette='colorblind')
plt.xticks(rotation=270)
plt.title("Rainfall Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Rainfall")
plt.show()

#Visualization the dataset for Potassium Vs Crop type
K1 = X.iloc[:,2]
plt.bar(y,K1,color="Red")
plt.xticks(rotation=270)
plt.title("Potassium Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Potassium")
plt.show()

#Visualization the dataset for Nitrogen Vs Crop type
N1 = X.iloc[:,0]
plt.bar(y,N1,color="Green")
plt.xticks(rotation=270)
plt.title("Nitrogen Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("Nitrogen")
plt.show()

#Visualization the dataset for Phosphorus Vs Crop type
P1 = X.iloc[:,1]
plt.bar(y,P1,color="Magenta")
plt.xticks(rotation=270)
plt.title("phosphorus Vs Crop type")
plt.xlabel("Crop type")
plt.ylabel("phosphorus")
plt.show()

# Encoding the data (Lebel Encoding)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y)
#Crop_encoding = {i : label for i, label in enumerate(labelencoder.classes_)}
#Crop_encoding

crop_encoded_value = pd.DataFrame(y1)
crop_encoded_value.columns=['Croptype']
frame = [X,crop_encoded_value]
df = pd.concat(frame, axis=1, join='inner')

# Correlation Matrix for Visualization the Dataset
sns.heatmap(df.corr(),annot=True)
plt.show()

# Visualization the data for balancing
unique_y = np.unique(y,return_counts=True)
V = np.array(unique_y).T
Values = pd.DataFrame(V)

C = Values.iloc[:,0]
V = Values.iloc[:,1]

sns.barplot(x=C,y=V,data=dataset,palette='bright')
plt.title("Data balancing")
plt.xlabel("Crops type")
plt.ylabel("No of crops")
plt.xticks(rotation=270)
plt.show()

# Spliting the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = .2, random_state = 42)
#print(y_test.max())

# Train the model using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(criterion = 'gini', random_state = 42)
decisiontree.fit(X_train, y_train)

# Predict the Output Value by using Decision Tree
Y_pred = decisiontree.predict(X_test)

accu = []
model = []

# Accuracy Calculation of Decision Tree
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, Y_pred)*100
accu = [accuracy]
model = ["Dicision Tree"]
print("Accuracy of Decision Tree is :",accuracy)

# Data Visualization: Actual Value Vs Predicted Value (Decision Tree)
from sklearn.metrics import confusion_matrix
f, ax = plt.subplots(figsize=(15,10))
con_mat = confusion_matrix(y_test,Y_pred)
sns.heatmap(con_mat,annot=True,ax=ax)
plt.title("Actual Value Vs Predicted Value")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()

# Train the model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 42)
randomforest.fit(X_train, y_train)

# Predict the Output Value
y_pred=randomforest.predict(X_test)

# Accuracy Calculation of Random Forest
from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_pred)*100
accu.append(accuracy1)
model.append("Random Forest")
print("Accuracy of Random Forest is :",accuracy1)

# Data Visualization: Actual Value Vs Predicted Value (Random Forest)
from sklearn.metrics import confusion_matrix
f, ax = plt.subplots(figsize=(15,10))
cn_mt = confusion_matrix(y_test,y_pred)
sns.heatmap(cn_mt,annot=True,ax=ax,cmap="Wistia")
plt.title("Actual Value Vs Predicted Value")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()

# Train the model using K Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict the Output Value
pred_values = knn.predict(X_test)

# Accuracy calculation of K Nearest Neighbor
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(y_test, pred_values)*100
accu.append(accuracy2)
model.append("K Nearest Neighbor")
print("Accuracy of K Nearest Neighbor is :",accuracy2)

# Data Visualization: Actual Value Vs Predicted Value (K Nearest Neighbor)
from sklearn.metrics import confusion_matrix
f, ax = plt.subplots(figsize=(15,10))
cn_mt = confusion_matrix(y_test, pred_values)
sns.heatmap(cn_mt,annot=True,ax=ax,cmap="Blues")
plt.title("Actual Value Vs Predicted Value")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()

# Hyperparameter Tuning for K Nearest Neighbor
mean_accu = []
for i in range(1,51):  
    knn1 = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    pred_values1= knn1.predict(X_test)
    mean_accu.append(accuracy_score(y_test, pred_values1))
#mean_accu

# Visualizatoin the Accuracy wrt Number of Neighbors
#plt.figure(figsize = (20,15))
#plt.xticks([int(i) for i in range(1,51)])
plt.plot(range(1,51), mean_accu)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()

# Finding the optimum parameter
parameter = { 'n_neighbors' : [12,13,14,15,16,17,18],
               'weights' : ['uniform','distance']}

from sklearn.model_selection import GridSearchCV
gridsearchcv = GridSearchCV(KNeighborsClassifier(), parameter, n_jobs = -1)

# Train the dataset by GridSearchCV
grid_fit = gridsearchcv.fit(X_train, y_train)

grid_fit.best_score_

grid_fit.best_params_

# Fit the data in KNeighborClasifier by the tuned parameter
knn2 = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
knn2.fit(X_train, y_train)

# Predict the value by tuned parameter
y_pred_values = knn2.predict(X_test)

# Accuracy calculation by tuned parameter
from sklearn.metrics import accuracy_score
Acc = accuracy_score(y_test, y_pred_values)*100
print("Accuracy of K Nearest Neighbor by tuned parameter is :",Acc)

# Train the model by Logistic Regression
from sklearn.linear_model import LogisticRegression
model2= LogisticRegression(random_state = 0)
model2.fit(X_train,y_train)

# Predict the output by Logictic Regression
Y_pred1 = model2.predict(X_test)

# Accuracy Calculation of Logictic Regression
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,Y_pred1)*100
accu.append(score)
model.append("Logistic Regression")
print("Accuracy of Logictic Regression :",score)

# Data Visualization: Actual Value Vs Predicted Value (Logictic Regression)
from sklearn.metrics import confusion_matrix
f, ax = plt.subplots(figsize=(15,10))
cn_mt = confusion_matrix(y_test, Y_pred1)
sns.heatmap(cn_mt,annot=True,ax=ax,cmap="Greens")
plt.title("Actual Value Vs Predicted Value")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()

# Accuracy Comparison
#scale = np.arange(0.95,1.0)
#sns.histplot(x = model,hue = accu,shrink=1,palette=['r','g'],log_scale=True)
plt.plot(model,accu,'r*')
plt.plot(model,accu)
plt.grid(alpha=0.7)
plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# Prediction of Crop by user input
'''a=float(input('Enter the Nitrogen value in soil: '))
b=float(input('Enter the Phosphorus value in soil: '))
c=float(input('Enter the Potassium value in soil: '))
d=float(input('Enter the Temperature: '))
e=float(input('Enter the Humidity: '))
f=float(input('Enter the ph level of the soil: ')) 
g=float(input('Enter the rainfall: '))

P = int(randomforest.predict([[a,b,c,d,e,f,g]]))
print(P)
if P==0:
  print("Predicted crop is: apple")
elif P==1:
  print("Predicted crop is: banana")
elif P==2:
  print("Predicted crop is: blackgram")
elif P==3:
  print("Predicted crop is: chickpea")
elif P==4:
  print("Predicted crop is: coconut")
elif P==5:
  print("Predicted crop is: coffee")
elif P==6:
  print("Predicted crop is: cotton")
elif P==7:
  print("Predicted crop is: grapes")
elif P==8:
  print("Predicted crop is: jute")
elif P==9:
  print("Predicted crop is: kidneybeans")
elif P==10:
  print("Predicted crop is: lentil")
elif P==11:
  print("Predicted crop is: maize")
elif P==12:
  print("Predicted crop is: mango")
elif P==13:
  print("Predicted crop is: mothbeans")
elif P==14:
  print("Predicted crop is: mungbean")
elif P==15:
  print("Predicted crop is: muskmelon")
elif P==16:
  print("Predicted crop is: orange")
elif P==17:
  print("Predicted crop is: papaya")
elif P==18:
  print("Predicted crop is: pigeonpeas")
elif P==19:
  print("Predicted crop is: pomegranate")
elif P==20:
  print("Predicted crop is: rice")
elif P==21:
  print("Predicted crop is: watermelon")'''



'''Y = np.unique(l,return_counts=True)
Values = np.array(Y).T
A = pd.DataFrame(Values)

C = A.iloc[:,0]
V = A.iloc[:,1]

P = C.tolist()
Q = V.tolist()

plt.bar(P,Q,color="Pink")
plt.xticks(rotation=270)
plt.show()


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(l)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size = .2, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Error Calculation
from sklearn import metrics
print("Error :",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Accuracy Detectoin
accuracy = (regressor.score(X_test,y_pred))
print("Accuracy(R^2 score) =",accuracy)

u1 = (y_test)**2
acc1 = np.sqrt(np.mean(u1))
print(acc1)

u2 = (y_pred)**2
acc2 = np.sqrt(np.mean(u2))
print(acc2)

Acc1 = (acc2/acc1)*100
print(Acc1)

x1 = np.mean(y_test)
print(x1)
x2 = np.mean(y_pred)
print(x2)

Acc2 = (x2/x1)*100
print(Acc2)


print(C)'''





 