
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing  import LabelEncoder
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
data=pd.read_excel('car_data1.xlsx')

print(data.head())

data.info()
method=1
if method==0:
    for i in data.columns:
        print(data[i].unique(),"\t", data[i].nunique())
    for i in data.columns:
        print(data[i].value_counts())
    sns.countplot(x="class", data=data)
    plt.show()

if method==1:
    for i in data.columns[:-1]:
        plt.figure(figsize=(12,6))
        plt.title("for feature '%s'"%i)
        sns.countplot(x=data[i],hue=data['class'])
    plt.show()

#chuyển đổi dữ liệu chuỗi thành số
if method==2:
    data1=data
    le = LabelEncoder()
    for i in data.columns:
        data1[i]=le.fit_transform(data1[i])
    print(data1.head())

    fig=plt.figure(figsize=(10,6))
    sns.heatmap(data.corr('pearson'),annot=True)
    plt.show()

#Random Forest
if 1:
    #dữ liệu cột X
    X=data[data.columns[:-1]]
    #dữ liệu dòng y
    y=data['class']
    X.head(2)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train.head()
    from sklearn.model_selection import learning_curve,cross_val_score,validation_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    #cho dữ liệu học
    rfc=RandomForestClassifier(n_jobs=-1,random_state=51)
    rfc.fit(X_train,y_train)
    #in accuracy và f1_score
    print(rfc.score(X_test,y_test))
    print(f1_score(y_test,rfc.predict(X_test),average='macro'))

    #Dự đoán kết quả tập thử nghiệm
    y_pred = rfc.predict(X_test)

    #In Confusion Matrix và cắt nó thành 4 phần
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm)
    print(classification_report(y_test, y_pred))
    
if 2:
    #Bây giờ, hãy kiểm tra ảnh hưởng của n_estimators đối với mô hình.
    param_range=[10,25,50,100]
    curve=validation_curve(rfc,X_train,y_train,cv=3,param_name='n_estimators',
        param_range=param_range,n_jobs=-1)
    train_score=[curve[0][i].mean() for i in range (0,len(param_range))]
    test_score=[curve[1][i].mean() for i in range (0,len(param_range))]
    fig=plt.figure(figsize=(6,8))
    plt.plot(param_range,train_score)
    plt.plot(param_range,test_score)
    plt.xticks=param_range
if 3:
    #Bây giờ, hãy kiểm tra cách mô hình phù hợp với các giá trị khác nhau của 'max_features'.
    param_range=range(1,len(X.columns)+1)
    curve=validation_curve(RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=51),X_train,y_train,cv=5,
        param_name='max_features',param_range=param_range,n_jobs=-1)
    train_score=[curve[0][i].mean() for i in range (0,len(param_range))]
    test_score=[curve[1][i].mean() for i in range (0,len(param_range))]
    fig=plt.figure(figsize=(6,8))
    plt.plot(param_range,train_score)
    plt.plot(param_range,test_score)
    plt.xticks=param_range

if 4:
    #chia GridSearch để tìm feature tốt nhất trong tập dữ liệu
    param_grid={'criterion':['gini','entropy'],
               'max_depth':[2,5,10,20],
               'max_features':[2,4,6,'auto'],
               'max_leaf_nodes':[2,3,None],}
    grid=GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=51),
                  param_grid=param_grid,cv=10,n_jobs=-1)
    grid.fit(X_train,y_train)
    print(grid.best_params_)
    print(grid.best_score_)

if 5:
    #Learning Curve
    lc=learning_curve(RandomForestClassifier(n_estimators=50,criterion='entropy',max_features=6,max_depth=10,random_state=51,
                                             max_leaf_nodes=None,n_jobs=-1,),X_train,y_train,cv=5,n_jobs=-1)
    size=lc[0]
    train_score=[lc[1][i].mean() for i in range (0,5)]
    test_score=[lc[2][i].mean() for i in range (0,5)]
    fig=plt.figure(figsize=(12,8))
    plt.plot(size,train_score)
    plt.plot(size,test_score)

    print(X.columns)
    print(rfc.feature_importances_)
    feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_scores
    sns.barplot(x=feature_scores, y=feature_scores.index)
    # Add labels to the graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    # Add title to the graph
    plt.title("Visualizing Important Features")
    # Visualize the graph
    plt.show()

if 6:
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[['buying', 'maint', 'persons', 'lug_boot', 'safety']],
        y, test_size=0.3, random_state=42)
    rfc1=RandomForestClassifier(n_estimators=50,criterion='entropy',max_features=5,max_depth=10,random_state=51,
        max_leaf_nodes=None,n_jobs=-1)
    rfc1.fit(X_train1,y_train1)
    rfc1.score(X_test1,y_test1)
