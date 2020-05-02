
import pandas as pd

desired_width=420
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',15) ##this is to view the output in a wider form


"""-----Loading data from Csv to Dataframe----------------"""
df = pd.read_csv('titanics.csv')
df['Sex'].replace(to_replace=['male','female'], value=[0,1], inplace=True) ##conevrt alpha values to numerical values
ndf = df[['Pclass','Sex','Age','SibSp','Parch']] #take only columns which have an impact
avg_age=(ndf['Age'].mean())  ## we found age in some rows are missing.Hence we assume the avg values

ndf['Age'].fillna(value=avg_age,inplace = True)##we fill missing values with avg value

y= df['Survived'] #this our target value


"""----------Splitting training and testing data----------------"""
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(ndf).transform(ndf) #normalizing the values to fit in X variable


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=4) ##we take 20%data for testing and rest for Training

"""-------------Training using Support Vector Method model------------------"""
from sklearn import svm

v = svm.SVC(kernel='rbf') #SVM with rbf kernel is uesd
v.fit(xtrain,ytrain)  ##fitting values to the model
yhat= v.predict(xtest)   ## predicting training set

from sklearn import metrics
print("SVM Accuracy: ", metrics.accuracy_score(ytest, yhat)) ##evaluation using inbuilt methods

"""------------------Training using logistic regresiion-------------------------"""
from sklearn.linear_model import LogisticRegression
bsf = LogisticRegression(C=0.01,solver='liblinear').fit(xtrain,ytrain)
dsr =bsf.predict(xtest)
print("lr Accuracy: ", metrics.accuracy_score(ytest, dsr))


"""------------------testing=--------------------------------"""
mdf = pd.read_csv('test1.csv')
mdf['Sex'].replace(to_replace=['male','female'], value=[0,1], inplace=True)
testf = mdf[['Pclass','Sex','Age','SibSp','Parch']]
nb=[]
new_avg=testf["Age"].mean()

testf['Age'].fillna(value=new_avg,inplace = True)

X1 = preprocessing.StandardScaler().fit(testf).transform(testf)
nb= v.predict(X1)
nedf = mdf
nedf.insert(1,"Survived",nb)



"""--------------end--------------------"""


