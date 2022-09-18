
import pandas as pd
path="./dataset/iris.csv"
a=pd.read_csv(path,sep=',')
a['Species']=a['Species'].replace('Iris-setosa','0').replace('Iris-versicolor','1')\
    .replace('Iris-virginica','2').astype('int32')

train_data=a.sample(frac=0.8,random_state=0)
test_data=a[~a.index.isin(train_data.index)]
train_data.to_csv(path+'train_data.csv',index=False)
test_data.to_csv(path+'test_data.csv',index=False)
