# Pandas Function

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-1.png)

Python is a great language for doing data analysis, primarily because of the fantastic ecosystem of data-centric python packages. Pandas is one of those packages and makes importing and analyzing data much easier.

## _PreRequisites_ 

Importing the Pandas Module

``` python
import pandas as pd 
```

**DataFrame** 

Dataframe is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. It can be thought of as a dict-like container for Series objects. This is the primary data structure of the Pandas.


## 1. Read_csv



**read_csv** is an important pandas function to read csv files and do operations on it.



| **Parameter**	|	**Use**| 
|----|---|
|filepath_or_buffer |	URL or Dir location of file|
| sep		|	Stands for separator, default is ‘, ‘ as in csv|
|index_col	|	Makes passed column as index instead |
|header	 |       	Makes passed row/s[int/int list] as header|
|use_cols|Only uses the passed col[string list] to make data frame|
|squeeze	|	If true and only one column is passed, returns pandas series|
skiprows|	Skips passed rows in new data frame|

```python
data = pd.read_csv("filename.csv")
```

---

## 2. Head and Tail



_head()_ method is used to return top n (5 by default) rows of a data frame or series

> Syntax : Dataframe.head(n).

Parameters: (optional) n is integer value, number of rows to be returned.

Return: Dataframe with top n rows .
```python
data.head()
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-33.png)

**TAIL**




_tail()_ method is used to return bottom n (5 by default) rows of a data frame or series.

> Syntax : Dataframe.tail(n)

Parameters: (optional) n is integer value, number of rows to be returned.

Return: Dataframe with bottom n rows .

```python
data.tail()
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-34.png)

---

## 3. Info()



_dataframe.info()_ function is used to get a concise summary of the dataframe. It comes really handy when doing exploratory analysis of the data. To get a quick overview of the dataset we use the dataframe.info() function.

> Syntax : DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)

```python
data.info()
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-35.png)

---

## 4. Dtypes



DataFrame.types attribute returns the dtypes in the DataFrame. It returns a Series with the data type of each column.

> Syntax : DataFrame.dtypes

Parameter : None

Returns : dtype of each column

```python
data.dtypes
data
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-44.png)

---

## 5. Describe



_describe()_ is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values. When this method is applied to a series of strings, it returns a different output which is shown in the examples below.

> Syntax : DataFrame.describe(percentiles=None, include=None, exclude=None)

Return type : Statistical summary of data frame.

```python
data.describe()
```

Output :
 
![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-45.png)

---

## 6. Size and Shapes



Pandas .size and .shape are used to return size and shape of data frames and series.

> Syntax : dataframe.size

Return : Returns size of dataframe/series which is equivalent to total number of elements.   

```python
data.size
```

Output :

`10692`



> Syntax : dataframe.shape

Return : Returns tuple of shape (Rows, columns) of dataframe/series

```python
data.shape
```
Output :

`(891,12)`

---

## 7. Sample



_sample()_ is used to generate a sample random row or column from the function caller data frame.

> Syntax : DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)

Return type : New object of same type as caller

```python
data.sample(n=1)
```

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-42.png)

---

## 8. Isnull()



column is checked for NULL values and a boolean series is returned by the isnull() method which stores True for ever NaN value and False for a Not null value.

>Syntax: Pandas.isnull(“DataFrameName”) or DataFrame.isnull()

Parameters: Object to check null values 

Return Type: Dataframe of Boolean values which are True for NaN values 

```python
data.isnull()
```
Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-46.png)
---

## 9. Isna()



Pandas dataframe.isna() function is used to detect missing values. It return a boolean same-sized object indicating if the values are NA. NA values(None or numpy.NaN) gets mapped to True values.

>Syntax: DataFrame.isna()

Returns: Mask of bool values for each element in DataFrame that indicates whether an element is an NA value or not.

```python
data.isna()
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-47.png)

---

## 10. Isnull().sum()




isnull().sum()- Returns the number of missing values in the data set.

example:
>Syntax  .isna().sum()   # or s.isnull().sum() for older pandas versions 

```python
data.isnull().sum()
```

Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-48.png)

---

## 11. nunique()




The function return number of unique elements in the object. It returns a value which is the count of all the unique values in the Index. By default the NaN values are not included in the count. If dropna parameter is set to be False then it includes NaN value in the count.

>Syntax: Index.nunique(dropna=True)

Parameters :
dropna : Don’t include NaN in the count.

Returns :int

```python
data.nunique
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-50.png)


---

## 12. Index and Column




Immutable sequence used for indexing and alignment. The basic object storing axis labels for all pandas objects.

> Syntax:  pandas.Index(data=None, dtype=None, copy=False, name=None, tupleize_cols=True, **kwargs)

An Index instance can only contain hashable objects

```python
data.index 
```

Output :

`RangeIndex(start=0, stop=891, step=1)`




Pandas DataFrame.columns attribute return the column labels of the given Dataframe.

> Syntax: DataFrame.columns

Parameter : None

Returns : column names 

```python
data.columns
```

Output :

`Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')`

---

## 13. Memory Usage

Pandas dataframe.memory_usage() function return the memory usage of each column in bytes. The memory usage can optionally include the contribution of the index and elements of object dtype. This value is displayed in DataFrame.info by default.

>Syntax: DataFrame.memory_usage(index=True, deep=False)

Parameters : 
index : Specifies whether to include the memory usage of the DataFrame’s index in returned Series. If index=True the memory usage of the index the first item in the output. 
deep : If True, introspect the data deeply by interrogating object dtypes for system-level memory consumption, and include it in the returned values.

Returns : A Series whose index is the original column names and whose values is the memory usage of each column in bytes 

```python
data.memory_usage()
```

Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-51.png)
---

## 14. nlargest and nsmallest



nsmallest() method is used to get n least values from a data frame or a series.

>Syntax : DataFrame.nsmallest(n, columns, keep=’first’)

```python
df = data.nsmallest(5,'Fare')
df
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-22.png)

**nlargest()**



nlargest() method is used to get n highest values from a data frame or a series.

> Syntax : DataFrame.nlargest(n, columns, keep=’first’)

Output : 

```python
df = data.nlargest(5,'Fare')
df
```
Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-24.png)

---

## 15. Loc and iloc



loc() and iloc() are used in slicing of data from the Pandas DataFrame. They help in the convenient selection of data from the DataFrame. They are used in filtering the data according to some conditions.


| loc |  iloc  |
|---|--|
|Access a group of rows and columns by label(s) or a boolean array. | Purely integer-location based indexing for selection by position.|

> Syntax : df.loc[row_indexer,column_indexer] 

```python
df = data.loc[10:15,['Fare']]
df
```

Output :



**iloc**


```python
df = data.iloc[3:7,:5]
df
```

Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-26.png)

---

## 16. Slicing



Slicing using the [] operator selects a set of rows and/or columns from a DataFrame. To slice out a set of rows, you use the following.

> syntax: data[start:stop]

```python
df = data[1:6]
df
```
Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-27.png)

---

## 17. Groupby



_groupby()_ function is used to split the data into groups based on some criteria.

> Syntax : DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, **kwargs)

Returns : DataFrameGroupBy

```python
df = data[['Fare','Age','Survived']].groupby(['Fare']).mean()
df
```
Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-28.png)

---

## 18. Sorting



_sort_values()_ function sorts a data frame in Ascending or Descending order of passed Column. It's different than the sorted Python function since it cannot sort a data frame and particular column cannot be selected.

> Syntax : DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)

Returns : DataFrame or None

```python
df = data.sort_index(axis = 1, ascending = True)
df
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-29.png)

```python
data.sort_index(axis = 1, ascending = False)
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-30.png)

```python
df = data.sort_values(by='Fare')
df
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-31.png)

---

## 19. Dropna

The dropna() function is used to remove missing values. Determine if rows or columns which contain missing values are removed



> Syntax : DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

Returns : DataFrame or None

```python
df = ['Fare'] 
data.drop(df, axis = 1, inplace = True)
data
```
Output :

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-32.png)

---

## 20. Query



> Syntax : DataFrame.query(expr, inplace=False, **kwargs)

Returns : DataFrame or None

_query()_ using “dot syntax”. Basically, type the name of the DataFrame you want to subset, then type a “dot”, and then type the name of the method --> query()

```python
data.query('18 < Age < 23')[:10]
```

Output : 

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-52.png)

---

## 21. Min(), Max(), Mean()



_min()_ function returns the minimum of the values in the given object. If the input is a series, the method will return a scalar which will be the minimum of the values in the series.

> Syntax : DataFrame.min(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)[source]

Returns : Series or DataFrame (if level specified)

```python
data['Age'].min()
```

Output :

` 0.42 `



max() function returns index of first occurrence of maximum over requested axis. While finding the index of the maximum value across any index, all NA/null values are excluded.

> Syntax : DataFrame.max(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)[source]

Returns : Series or DataFrame (if level specified)

```python
data['Age'].max()
```

Output : 

` 80 `



_mean()_ function is used to return the mean of the values for the requested axis. If we apply this method on a Series object, then it returns a scalar value, which is the mean value of all the observations in the dataframe.

> Syntax : DataFrame.mean(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)[source]

Returns : Series or DataFrame (if level specified)

```python
data['Age'].mean()
```

Output :

` 29.69911764705882 `


---
---

## Thankyou DevIncept

## Content Created By
- Nagashree M S
- Prajakta
- Rammya Dharshini K

![](https://github.com/rammya29/Intern-Work/blob/main/int-py-8/Pandas%20Function/Images/Img-2.png)
