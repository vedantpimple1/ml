import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns


```python
df=pd.read_csv('temperatures.csv')
```


```python
df.head()
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
      <th>YEAR</th>
      <th>JAN</th>
      <th>FEB</th>
      <th>MAR</th>
      <th>APR</th>
      <th>MAY</th>
      <th>JUN</th>
      <th>JUL</th>
      <th>AUG</th>
      <th>SEP</th>
      <th>OCT</th>
      <th>NOV</th>
      <th>DEC</th>
      <th>ANNUAL</th>
      <th>JAN-FEB</th>
      <th>MAR-MAY</th>
      <th>JUN-SEP</th>
      <th>OCT-DEC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1901</td>
      <td>22.40</td>
      <td>24.14</td>
      <td>29.07</td>
      <td>31.91</td>
      <td>33.41</td>
      <td>33.18</td>
      <td>31.21</td>
      <td>30.39</td>
      <td>30.47</td>
      <td>29.97</td>
      <td>27.31</td>
      <td>24.49</td>
      <td>28.96</td>
      <td>23.27</td>
      <td>31.46</td>
      <td>31.27</td>
      <td>27.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1902</td>
      <td>24.93</td>
      <td>26.58</td>
      <td>29.77</td>
      <td>31.78</td>
      <td>33.73</td>
      <td>32.91</td>
      <td>30.92</td>
      <td>30.73</td>
      <td>29.80</td>
      <td>29.12</td>
      <td>26.31</td>
      <td>24.04</td>
      <td>29.22</td>
      <td>25.75</td>
      <td>31.76</td>
      <td>31.09</td>
      <td>26.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1903</td>
      <td>23.44</td>
      <td>25.03</td>
      <td>27.83</td>
      <td>31.39</td>
      <td>32.91</td>
      <td>33.00</td>
      <td>31.34</td>
      <td>29.98</td>
      <td>29.85</td>
      <td>29.04</td>
      <td>26.08</td>
      <td>23.65</td>
      <td>28.47</td>
      <td>24.24</td>
      <td>30.71</td>
      <td>30.92</td>
      <td>26.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1904</td>
      <td>22.50</td>
      <td>24.73</td>
      <td>28.21</td>
      <td>32.02</td>
      <td>32.64</td>
      <td>32.07</td>
      <td>30.36</td>
      <td>30.09</td>
      <td>30.04</td>
      <td>29.20</td>
      <td>26.36</td>
      <td>23.63</td>
      <td>28.49</td>
      <td>23.62</td>
      <td>30.95</td>
      <td>30.66</td>
      <td>26.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1905</td>
      <td>22.00</td>
      <td>22.83</td>
      <td>26.68</td>
      <td>30.01</td>
      <td>33.32</td>
      <td>33.25</td>
      <td>31.44</td>
      <td>30.68</td>
      <td>30.12</td>
      <td>30.67</td>
      <td>27.52</td>
      <td>23.82</td>
      <td>28.30</td>
      <td>22.25</td>
      <td>30.00</td>
      <td>31.33</td>
      <td>26.57</td>
    </tr>
  </tbody>
</table>
</div>




```python
x=df[['YEAR']]
y=df['ANNUAL']
```


```python
x.shape
```




    (117, 1)




```python
lr=LinearRegression()
```


```python
lr.fit(x,y)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
lr.coef_
```




    array([0.01312158])




```python
lr.intercept_
```




    3.4761897126187016




```python
y_pred = lr.predict(x)
```


```python
mae=mean_absolute_error(y,y_pred)
mse=mean_squared_error(y,y_pred)
r2score=r2_score(y,y_pred)
```


```python
print(f"Mean Squared Error:{mse : .4f}")
print(f"Mean Absolute Error:{mae: .4f}")
print(f"R2_Score:{r2score: .4f}")
```

    Mean Squared Error: 0.1096
    Mean Absolute Error: 0.2254
    R2_Score: 0.6418
    


```python

plt.title('Temp PLot of INdia')
plt.xlabel('Year')
plt.ylabel('Annual Avg Temp')
plt.scatter(x,y,label="Actua Values",marker=".")
plt.plot(x,y_pred,label="Predictred Values",color='red')
plt.legend()
```




    <matplotlib.legend.Legend at 0x74b0a8332f50>




    
![png](output_12_1.png)
    



```python
sns.regplot(x='YEAR',y='ANNUAL',data=df)
```




    <Axes: xlabel='YEAR', ylabel='ANNUAL'>




    
![png](output_13_1.png)
    



```python

```
