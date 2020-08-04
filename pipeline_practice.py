
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/usedcars.csv')

df.head()

df.info()

df.dtypes

df.year.value_counts().plot(kind = 'bar')

df.color.value_counts().plot(kind ='bar')

df.mileage.value_counts()

X= df.drop('price',axis=1)
y= df.price
categorical_features = df.columns[df.dtypes == 'O']
numerical_features = ['year',  'mileage']

print('categorical Featues are :', categorical_features.to_list())
print("numerciacl features are :",numerical_features)

df.price.plot(kind='hist',)

cat_transform = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
                                ('encoder',OneHotEncoder(handle_unknown='ignore'))])
num_transform = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

processor = ColumnTransformer(transformers=[('cat',cat_transform,categorical_features),
                              ('num',num_transform,numerical_features)])

logreg = Pipeline(steps=[('processor',processor),('linear',LinearRegression())])

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state= 42)

logreg.fit(X_train,y_train)

print("model score: %.3f" % logreg.score(X_test, y_test))



