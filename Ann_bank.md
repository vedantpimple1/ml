```python
import pandas as pd
```


```python
df=pd.read_csv('/content/ANN_Bank_Marketing.csv')
```


```python
df
```





  <div id="df-95b5ed9a-ba30-42c2-91df-90d95ca2be7c" class="colab-df-container">
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>NaN</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>NaN</td>
      <td>single</td>
      <td>NaN</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>NaN</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
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
      <th>45206</th>
      <td>51</td>
      <td>technician</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>825</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>977</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45207</th>
      <td>71</td>
      <td>retired</td>
      <td>divorced</td>
      <td>primary</td>
      <td>no</td>
      <td>1729</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>456</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45208</th>
      <td>72</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>5715</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>1127</td>
      <td>5</td>
      <td>184</td>
      <td>3</td>
      <td>success</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45209</th>
      <td>57</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>668</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>17</td>
      <td>nov</td>
      <td>508</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45210</th>
      <td>37</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2971</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>361</td>
      <td>2</td>
      <td>188</td>
      <td>11</td>
      <td>other</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>45211 rows × 17 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95b5ed9a-ba30-42c2-91df-90d95ca2be7c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-95b5ed9a-ba30-42c2-91df-90d95ca2be7c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95b5ed9a-ba30-42c2-91df-90d95ca2be7c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-63b36a87-fa23-43c5-9d2d-48e4d5d62971">
      <button class="colab-df-quickchart" onclick="quickchart('df-63b36a87-fa23-43c5-9d2d-48e4d5d62971')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-63b36a87-fa23-43c5-9d2d-48e4d5d62971 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_51a1fb83-cfe0-4b10-9d21-5934c1881211">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_51a1fb83-cfe0-4b10-9d21-5934c1881211 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
from sklearn.preprocessing import LabelEncoder
```


```python
le=LabelEncoder()
```


```python
for col in df.columns:
  df[col]=le.fit_transform(df[col])
```


```python
df.dtypes
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>job</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>marital</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>education</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>default</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>balance</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>housing</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>loan</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>contact</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>month</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>campaign</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>pdays</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>previous</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>poutcome</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>y</th>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




```python
x=df.drop('y',axis=1)
y=df['y']
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


```python
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
```


```python
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
```


```python
model=Sequential()
```


```python

model.add(Dense(16,input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))
```

    /usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    


```python
history=model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```


```python
history=model.fit(X_train,y_train,epochs=100,batch_size=100,validation_data=(X_test,y_test))
plot_model(model,show_shapes=True,show_layer_names=True)
```

    Epoch 1/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9054 - loss: 0.2134 - val_accuracy: 0.8994 - val_loss: 0.2344
    Epoch 2/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9057 - loss: 0.2106 - val_accuracy: 0.9000 - val_loss: 0.2340
    Epoch 3/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9082 - loss: 0.2077 - val_accuracy: 0.8993 - val_loss: 0.2378
    Epoch 4/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9058 - loss: 0.2073 - val_accuracy: 0.8987 - val_loss: 0.2323
    Epoch 5/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9072 - loss: 0.2119 - val_accuracy: 0.8988 - val_loss: 0.2351
    Epoch 6/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9065 - loss: 0.2069 - val_accuracy: 0.8959 - val_loss: 0.2346
    Epoch 7/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9076 - loss: 0.2089 - val_accuracy: 0.8994 - val_loss: 0.2355
    Epoch 8/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9051 - loss: 0.2100 - val_accuracy: 0.9001 - val_loss: 0.2347
    Epoch 9/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9058 - loss: 0.2106 - val_accuracy: 0.8964 - val_loss: 0.2358
    Epoch 10/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9038 - loss: 0.2103 - val_accuracy: 0.8997 - val_loss: 0.2330
    Epoch 11/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9063 - loss: 0.2089 - val_accuracy: 0.8974 - val_loss: 0.2340
    Epoch 12/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9075 - loss: 0.2076 - val_accuracy: 0.8985 - val_loss: 0.2326
    Epoch 13/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9053 - loss: 0.2124 - val_accuracy: 0.9006 - val_loss: 0.2333
    Epoch 14/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9071 - loss: 0.2066 - val_accuracy: 0.8976 - val_loss: 0.2340
    Epoch 15/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9050 - loss: 0.2131 - val_accuracy: 0.9003 - val_loss: 0.2345
    Epoch 16/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 4ms/step - accuracy: 0.9055 - loss: 0.2107 - val_accuracy: 0.8998 - val_loss: 0.2328
    Epoch 17/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9084 - loss: 0.2047 - val_accuracy: 0.8976 - val_loss: 0.2347
    Epoch 18/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9075 - loss: 0.2087 - val_accuracy: 0.9001 - val_loss: 0.2335
    Epoch 19/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9069 - loss: 0.2104 - val_accuracy: 0.8999 - val_loss: 0.2352
    Epoch 20/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9086 - loss: 0.2073 - val_accuracy: 0.8991 - val_loss: 0.2332
    Epoch 21/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9045 - loss: 0.2096 - val_accuracy: 0.8997 - val_loss: 0.2334
    Epoch 22/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9077 - loss: 0.2039 - val_accuracy: 0.8983 - val_loss: 0.2345
    Epoch 23/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9079 - loss: 0.2061 - val_accuracy: 0.9003 - val_loss: 0.2330
    Epoch 24/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9079 - loss: 0.2079 - val_accuracy: 0.8987 - val_loss: 0.2339
    Epoch 25/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9049 - loss: 0.2100 - val_accuracy: 0.8997 - val_loss: 0.2337
    Epoch 26/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9079 - loss: 0.2056 - val_accuracy: 0.8994 - val_loss: 0.2335
    Epoch 27/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9065 - loss: 0.2057 - val_accuracy: 0.8998 - val_loss: 0.2345
    Epoch 28/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9049 - loss: 0.2129 - val_accuracy: 0.8990 - val_loss: 0.2327
    Epoch 29/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9068 - loss: 0.2103 - val_accuracy: 0.8986 - val_loss: 0.2349
    Epoch 30/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2091 - val_accuracy: 0.8995 - val_loss: 0.2329
    Epoch 31/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9045 - loss: 0.2099 - val_accuracy: 0.8998 - val_loss: 0.2336
    Epoch 32/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9070 - loss: 0.2071 - val_accuracy: 0.9001 - val_loss: 0.2320
    Epoch 33/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9065 - loss: 0.2085 - val_accuracy: 0.8983 - val_loss: 0.2342
    Epoch 34/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 6ms/step - accuracy: 0.9078 - loss: 0.2088 - val_accuracy: 0.8993 - val_loss: 0.2330
    Epoch 35/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9078 - loss: 0.2089 - val_accuracy: 0.9000 - val_loss: 0.2336
    Epoch 36/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9097 - loss: 0.2041 - val_accuracy: 0.9015 - val_loss: 0.2328
    Epoch 37/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9086 - loss: 0.2076 - val_accuracy: 0.8990 - val_loss: 0.2328
    Epoch 38/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9076 - loss: 0.2076 - val_accuracy: 0.8988 - val_loss: 0.2350
    Epoch 39/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2063 - val_accuracy: 0.8996 - val_loss: 0.2335
    Epoch 40/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9087 - loss: 0.2045 - val_accuracy: 0.9007 - val_loss: 0.2332
    Epoch 41/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9089 - loss: 0.2105 - val_accuracy: 0.8993 - val_loss: 0.2339
    Epoch 42/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9063 - loss: 0.2094 - val_accuracy: 0.9001 - val_loss: 0.2338
    Epoch 43/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9058 - loss: 0.2067 - val_accuracy: 0.9014 - val_loss: 0.2328
    Epoch 44/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9054 - loss: 0.2085 - val_accuracy: 0.9001 - val_loss: 0.2331
    Epoch 45/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2083 - val_accuracy: 0.9011 - val_loss: 0.2327
    Epoch 46/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9084 - loss: 0.2055 - val_accuracy: 0.9000 - val_loss: 0.2336
    Epoch 47/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9092 - loss: 0.2032 - val_accuracy: 0.8995 - val_loss: 0.2332
    Epoch 48/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9073 - loss: 0.2078 - val_accuracy: 0.9009 - val_loss: 0.2342
    Epoch 49/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9094 - loss: 0.2061 - val_accuracy: 0.9000 - val_loss: 0.2326
    Epoch 50/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9056 - loss: 0.2106 - val_accuracy: 0.9004 - val_loss: 0.2334
    Epoch 51/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9085 - loss: 0.2083 - val_accuracy: 0.9004 - val_loss: 0.2358
    Epoch 52/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9047 - loss: 0.2112 - val_accuracy: 0.8994 - val_loss: 0.2320
    Epoch 53/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9064 - loss: 0.2109 - val_accuracy: 0.9006 - val_loss: 0.2328
    Epoch 54/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9088 - loss: 0.2060 - val_accuracy: 0.9005 - val_loss: 0.2344
    Epoch 55/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9071 - loss: 0.2093 - val_accuracy: 0.9004 - val_loss: 0.2343
    Epoch 56/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9065 - loss: 0.2064 - val_accuracy: 0.9000 - val_loss: 0.2332
    Epoch 57/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9068 - loss: 0.2082 - val_accuracy: 0.8989 - val_loss: 0.2367
    Epoch 58/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9096 - loss: 0.2064 - val_accuracy: 0.8987 - val_loss: 0.2321
    Epoch 59/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9065 - loss: 0.2099 - val_accuracy: 0.8999 - val_loss: 0.2340
    Epoch 60/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9079 - loss: 0.2075 - val_accuracy: 0.8995 - val_loss: 0.2329
    Epoch 61/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9103 - loss: 0.2039 - val_accuracy: 0.8993 - val_loss: 0.2330
    Epoch 62/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9073 - loss: 0.2078 - val_accuracy: 0.8997 - val_loss: 0.2324
    Epoch 63/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9099 - loss: 0.2073 - val_accuracy: 0.9001 - val_loss: 0.2335
    Epoch 64/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9068 - loss: 0.2072 - val_accuracy: 0.8999 - val_loss: 0.2321
    Epoch 65/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2027 - val_accuracy: 0.8979 - val_loss: 0.2335
    Epoch 66/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9095 - loss: 0.2049 - val_accuracy: 0.8974 - val_loss: 0.2343
    Epoch 67/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9088 - loss: 0.2051 - val_accuracy: 0.8979 - val_loss: 0.2324
    Epoch 68/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2063 - val_accuracy: 0.8995 - val_loss: 0.2330
    Epoch 69/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9069 - loss: 0.2083 - val_accuracy: 0.8998 - val_loss: 0.2323
    Epoch 70/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9107 - loss: 0.2068 - val_accuracy: 0.9008 - val_loss: 0.2341
    Epoch 71/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9109 - loss: 0.2021 - val_accuracy: 0.9009 - val_loss: 0.2328
    Epoch 72/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9075 - loss: 0.2106 - val_accuracy: 0.9005 - val_loss: 0.2336
    Epoch 73/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2059 - val_accuracy: 0.8995 - val_loss: 0.2340
    Epoch 74/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9087 - loss: 0.2075 - val_accuracy: 0.8991 - val_loss: 0.2321
    Epoch 75/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9105 - loss: 0.2062 - val_accuracy: 0.8987 - val_loss: 0.2314
    Epoch 76/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9082 - loss: 0.2077 - val_accuracy: 0.8984 - val_loss: 0.2333
    Epoch 77/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9085 - loss: 0.2072 - val_accuracy: 0.8996 - val_loss: 0.2310
    Epoch 78/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9095 - loss: 0.2073 - val_accuracy: 0.9015 - val_loss: 0.2326
    Epoch 79/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9094 - loss: 0.2072 - val_accuracy: 0.9009 - val_loss: 0.2322
    Epoch 80/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9094 - loss: 0.2066 - val_accuracy: 0.9015 - val_loss: 0.2309
    Epoch 81/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9084 - loss: 0.2078 - val_accuracy: 0.8989 - val_loss: 0.2316
    Epoch 82/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.9104 - loss: 0.2030 - val_accuracy: 0.9016 - val_loss: 0.2326
    Epoch 83/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9078 - loss: 0.2067 - val_accuracy: 0.8998 - val_loss: 0.2321
    Epoch 84/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - accuracy: 0.9084 - loss: 0.2055 - val_accuracy: 0.8993 - val_loss: 0.2330
    Epoch 85/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9067 - loss: 0.2081 - val_accuracy: 0.9005 - val_loss: 0.2317
    Epoch 86/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9066 - loss: 0.2072 - val_accuracy: 0.9004 - val_loss: 0.2315
    Epoch 87/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9080 - loss: 0.2041 - val_accuracy: 0.8987 - val_loss: 0.2311
    Epoch 88/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9073 - loss: 0.2071 - val_accuracy: 0.8989 - val_loss: 0.2312
    Epoch 89/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9097 - loss: 0.2041 - val_accuracy: 0.8982 - val_loss: 0.2304
    Epoch 90/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 6ms/step - accuracy: 0.9069 - loss: 0.2106 - val_accuracy: 0.9018 - val_loss: 0.2322
    Epoch 91/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9078 - loss: 0.2075 - val_accuracy: 0.8979 - val_loss: 0.2330
    Epoch 92/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9071 - loss: 0.2048 - val_accuracy: 0.9012 - val_loss: 0.2317
    Epoch 93/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9087 - loss: 0.2081 - val_accuracy: 0.8989 - val_loss: 0.2301
    Epoch 94/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 3ms/step - accuracy: 0.9047 - loss: 0.2107 - val_accuracy: 0.8986 - val_loss: 0.2325
    Epoch 95/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9086 - loss: 0.2078 - val_accuracy: 0.9006 - val_loss: 0.2324
    Epoch 96/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9083 - loss: 0.2002 - val_accuracy: 0.8996 - val_loss: 0.2308
    Epoch 97/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9058 - loss: 0.2092 - val_accuracy: 0.8990 - val_loss: 0.2311
    Epoch 98/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 5ms/step - accuracy: 0.9080 - loss: 0.2064 - val_accuracy: 0.8980 - val_loss: 0.2352
    Epoch 99/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - accuracy: 0.9090 - loss: 0.2040 - val_accuracy: 0.8987 - val_loss: 0.2303
    Epoch 100/100
    [1m362/362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - accuracy: 0.9076 - loss: 0.2053 - val_accuracy: 0.8991 - val_loss: 0.2303
    




    
![png](output_15_1.png)
    




```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">272</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">136</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">72</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">9</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">32</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">136</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">72</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">9</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,216</span> (8.66 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">738</span> (2.88 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,478</span> (5.78 KB)
</pre>




```python
import matplotlib.pyplot as plt
```


```python

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7c209b703380>




    
![png](output_18_1.png)
    



```python

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7c20b4171970>




    
![png](output_19_1.png)
    



```python

```


```python

```


```python

```
