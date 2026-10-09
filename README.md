# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Andal, John Lester | 22-06394 | MEXE-4103 |
| Ramos, Mark Harry | 22-04113 | MEXE-4103 |

## Notebook links

| Chapter | Andal | Ramos |
|---|---|---|
| Ch1_2_3 | [Andal_Ch1_2_3](https://colab.research.google.com/drive/1JqWPtkRFbfcIdKXRJbOGyARjSB-hbo_P?usp=drive_link) | [Ramos Ch1_2_3](https://colab.research.google.com/drive/1SSskdLsOk4Zad_Hk6fV-qfWHWQcvtS6A) |
| Ch4 | [Andal_Ch4](https://colab.research.google.com/drive/1RHl9thSPiI_9JEmZrS7U5Oj2HJ4RtOWL?usp=drive_link) | [Ramos_Ch4](https://colab.research.google.com/drive/1DR7ortiALB8B0nO8nFEQGEKA-qxmDEqv) |
| Ch5 | [Andal_Ch5](https://colab.research.google.com/drive/1X6tLuBb34RiRdblMmaU3yY0mCgck3cLE?usp=drive_link) | [Ramos_Ch5](https://colab.research.google.com/drive/1P43cVEJcbV3bcGnBSijX84Su9YjgzTXn) |
| Ch6 | [Andal_Ch6](https://colab.research.google.com/drive/11zloPjh4CEg0Saag-OX6OLVg3HPdwECF?usp=drive_link) | [Ramos_Ch6](https://colab.research.google.com/drive/1ckhmwvPmijHeIwl0KdQgwgz6yxRkDTSh) |
| Ch7 | [Andal_Ch7](https://colab.research.google.com/drive/11gKpZZTUuGlyOopEddLQXa9e_jls1VfM?usp=drive_link) | [Ramos_Ch7](https://colab.research.google.com/drive/1-V6PNU7pmqie9poD9MbARCMPUHij7DZn) |
| Ch8 | [Andal_Ch8](https://colab.research.google.com/drive/1OONiWZBb81peW5Tc-JK5yH0PlYhe33N_?usp=drive_link) | [Ramos_Ch8](https://colab.research.google.com/drive/178qbFtRwRCjeJtPyR_ZYayklDcyFU56I) |
| Ch9 | [Andal_Ch9](https://colab.research.google.com/drive/115DrbnoRXPU83BeerZj3uYdPU6Abm2Uu?usp=drive_link) | [Ramos_Ch9](https://colab.research.google.com/drive/1k-5Ax0w4bTechrfhgrmlxV0sCLH202A-) |

## 📝 WHAT WE LEARNED

### ‣ Chapter 1-3 (Introduction to Preprocessing, Exploring and Cleaning Data)
We learned that raw datasets are almost always messy, incomplete, or filled with inconsistent formatting that must be cleaned before modeling. What surprised us was how missing values aren't just empty cells to be deleted—recklessly dropping them can destroy valuable context, so choosing between imputation strategies and dropping rows requires careful judgment.

### ‣ Chapter 4 (Transformation, Feature Engineering, and Encoding)
We learned how machine learning models require text and categories to be converted into numerical formats like One-Hot Encoding or Label Encoding. We're surprised to realize that assigning simple numbers to categories (like 1, 2, 3) can accidentally trick a model into thinking one category is "greater than" another, which is why creating dummy binary columns is necessary for non-ordinal features.

### ‣ Chapter 5 (Scaling and Normalization)
We learned that features measured on different scales (like age vs. income) can cause models to heavily favor columns with larger numbers. What surprised us most was that standardizing a feature doesn't change its underlying distribution shape, it simply adjusts the mean to 0 and standard deviation to 1 so every variable competes fairly.

### ‣ Chapter 6 (Outlier Detection)
We learned how extreme values can severely skew statistical metrics and distort machine learning predictions. What surprised us was discovering that outliers aren't automatically errors to be deleted, sometimes capping them at upper/lower thresholds preserves critical data volume without throwing off the model.

### ‣ Chapter 7 (	Feature Selection)
We learned that more data isn't always better, as redundant or irrelevant features can actually reduce model accuracy and increase computational cost. We were surprised by how wrapper methods like RFECV test actual combinations of variables interactively, whereas filter methods look at isolated statistical relationships like correlation.

### ‣ Chapter 8 (Constructing a Preprocessing Pipeline)
We learned how pipelines bundle cleaning, scaling, and transformation steps into a unified, reproducible sequence like a factory conveyor belt. What surprised us was how ColumnTransformer allows different feature types (like numerical vs. categorical) to pass through completely different preprocessing chains simultaneously within the same workflow.

### ‣ Chapter 9 (	Full Pipeline and Visualization)
We learned how to combine data cleaning, discretization, encoding, and visualization into an end-to-end workflow on the Titanic dataset. What surprised us was seeing how converting continuous numbers like age into discrete life-stage bins (Child, Adult, Elderly) made survival patterns immediately clear during visualization.

## ❌ ERRORS WE FOUND

### ‣ Chapter 3

**Issue:** Calling `inplace=True` on Series `fillna()` calls triggers `FutureWarning` in modern Pandas versions because chained assignment with inplace modification is deprecated.

**Wrong Version:**
```python
df['Year'].fillna(df['Year'].mean(), inplace=True)
df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)
```
**Right Version:**
```python
df['Year'] = df['Year'].fillna(df['Year'].mean())
df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])
```
---

### ‣ Chapter 7
**Issue:** df_2 contains only 7 rows of data. Using 5-fold cross-validation (cv=5) results in folds with fewer than 2 samples. This triggers continuous UndefinedMetricWarning: R^2 score is not well-defined with less than two samples warnings and yields inaccurate selection results.

<p></p>

**Wrong Version:**
```python
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(df_2.drop('final grade', axis=1), df_2['final grade'])
```

**Right Version:**
```python
from sklearn.feature_selection import RFE

# Use standard RFE without cross-validation for extremely small datasets
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(df_2.drop('final grade', axis=1), df_2['final grade'])
print(df_2.drop('final grade', axis=1).columns[selector.support_])
```
---

## 🤖 NOTE ON AI TOOLS
AI tools (ChatGPT/Gemini) were used as a learning assistant and thought partner throughout this notebook
* There are some parts of chapters we use AI about something as the description is too short for us to know and understand about it.
* We also use AI to find the format of the code to know where we can see and gather an answer

## REFERENCES

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
