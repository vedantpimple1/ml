```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,classification_report
```


```python
df=pd.read_csv('Admission_Predict (1).csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>396</td>
      <td>324</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>9.04</td>
      <td>1</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>396</th>
      <td>397</td>
      <td>325</td>
      <td>107</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>9.11</td>
      <td>1</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>330</td>
      <td>116</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>9.45</td>
      <td>1</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>312</td>
      <td>103</td>
      <td>3</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>8.78</td>
      <td>0</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>333</td>
      <td>117</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.66</td>
      <td>1</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 9 columns</p>
</div>




```python
df.shape
```




    (400, 9)




```python
df.drop(['Serial No.'], axis=1,inplace=True)
```


```python
df['Admit']=df['Chance of Admit '].apply(lambda x : 1 if x>=0.75 else 0)
```


```python
df.drop(['Chance of Admit '],axis=1,inplace=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>324</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>9.04</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>325</td>
      <td>107</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>9.11</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>330</td>
      <td>116</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>9.45</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>312</td>
      <td>103</td>
      <td>3</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>8.78</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>333</td>
      <td>117</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.66</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 8 columns</p>
</div>




```python
x=df.drop('Admit',axis=1)
y=df['Admit']
```


```python
sns.countplot(x=y)
```




    <Axes: xlabel='Admit', ylabel='count'>




    
![png](output_9_1.png)
    



```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42);
```


```python
dtc=DecisionTreeClassifier()
```


```python
dtc.fit(x_train,y_train)
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>




```python
y_pred=dtc.predict(x_test)
```


```python
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x71e578525a10>




    
![png](output_14_1.png)
    



```python
accuracy_score(y_test,y_pred)
```




    0.86




```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.81      0.87        59
               1       0.78      0.93      0.84        41
    
        accuracy                           0.86       100
       macro avg       0.86      0.87      0.86       100
    weighted avg       0.87      0.86      0.86       100
    
    


```python
plt.figure(figsize=(20,12))
plot_tree(dtc,fontsize=8,filled=True,rounded=True,class_names=['NA','AD']);
```


    
![png](output_17_0.png)
    



```python

```
