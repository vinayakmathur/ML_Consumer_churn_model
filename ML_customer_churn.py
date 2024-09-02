import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgbt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

df_churn = pd.read_csv("C:\\Users\\Vinayak Mathur\\Documents\\Churn_Data.csv")
#print(df_churn.info())
#print(df_churn.isnull().sum())
df_churn.drop(['Surname'],axis=1,inplace = True)
df_churn.drop(['Gender'],axis=1,inplace = True)
df_churn.drop(['RowNumber'],axis=1,inplace = True)
df_churn.drop(['CustomerId'],axis=1,inplace = True)

palette={"Germany": "#F0E100", "France": "#01295F", "Spain": "#F17105"}
palette_df=pd.DataFrame(palette.items(), columns=['Geography', 'Color'])


customers = pd.DataFrame(df_churn['Geography'].value_counts(normalize=False))

customers = customers.reset_index().rename(columns = {'index':'Geography','Geography':'Geography'})


customers=pd.merge(customers, palette_df, on=['Geography'],how='outer')
print(customers)
plt.pie(customers['count'], labels=customers['Geography'], colors=customers['Color'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Customers Geographical Locations")
#plt.show()
churners=df_churn[df_churn['Exited']==1]
nonchurners=df_churn[df_churn['Exited']==0]
churners_counta=pd.DataFrame(churners['Geography'].value_counts(normalize=False))
churners_counta=churners_counta.reset_index().rename(columns = {'index':'Geography','Geography':'Geography'})
churners_count=pd.merge(churners_counta, palette_df, on=['Geography'],how='outer')

nonchurners_counta=pd.DataFrame(nonchurners['Geography'].value_counts(normalize=False))
nonchurners_counta=nonchurners_counta.reset_index().rename(columns = {'index':'Geography','Geography':'Geography'})
nonchurners_count=pd.merge(nonchurners_counta, palette_df, on=['Geography'],how='outer')

print(sns.displot(data=churners, x=churners["Age"], hue='Geography',kde=True,height=7,palette=palette))
sns.displot(data=nonchurners, x=nonchurners["Age"], hue='Geography',kde=True,height=10,palette=palette)
sns.displot(df_churn, x=df_churn["EstimatedSalary"], hue="Exited",multiple="stack",height=10)
sns.displot(churners, x=churners["CreditScore"], hue="Geography", multiple="stack",height=10,palette=palette)
#plt.show()	
balance_df=pd.DataFrame(df_churn['Exited'].value_counts(normalize=False))
balance_df=balance_df.reset_index().rename(columns = {'index':'Exited','Exited':'Exited'})
#print(balance_df)

#print(df_churn['Geography'].unique())
le = LabelEncoder()
df_churn['Geography']=le.fit_transform(df_churn['Geography'])
le_geography_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_geography_mapping)

X = df_churn.drop("Exited", axis=1)
y = df_churn['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
xgb_churn = xgbt.XGBClassifier(gamma= 1.0,learning_rate= 0.15,max_depth= 7,
                                  n_estimators= 100)

xgb_churn.fit(X_train, y_train)
predictions = xgb_churn.predict(X_test)
accuracy_xgb = accuracy_score(y_test, predictions) * 100
print("The accuracy score of the XGBoost Classification Model is: ",accuracy_xgb)




sc = StandardScaler()
X_rftrain = sc.fit_transform(X_train)
X_rftest = sc.transform(X_test)
rfc = RandomForestClassifier(criterion ='gini', n_estimators=100,random_state = 10)
rfc.fit(X_rftrain, y_train)

rfc_pred_train = rfc.predict(X_rftrain)
rfc_pred_test = rfc.predict(X_rftest)
confusion_matrix(y_test, rfc_pred_test)
accuracy_rf=accuracy_score(y_test, rfc_pred_test)*100
print("The accuracy score of the Random Forest Classification Model is: ",accuracy_rf)
print(classification_report(y_test, rfc_pred_test))
