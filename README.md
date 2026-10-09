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
| Ch1_2_3 | [https://colab.research.google.com/drive/1JqWPtkRFbfcIdKXRJbOGyARjSB-hbo_P?usp=drive_link]() | [https://colab.research.google.com/drive/1SSskdLsOk4Zad_Hk6fV-qfWHWQcvtS6A?usp=sharing]() |
| Ch4 | [https://colab.research.google.com/drive/1RHl9thSPiI_9JEmZrS7U5Oj2HJ4RtOWL?usp=drive_link]() | [https://colab.research.google.com/drive/1DR7ortiALB8B0nO8nFEQGEKA-qxmDEqv?usp=sharing]() |
| Ch5 | [https://colab.research.google.com/drive/1X6tLuBb34RiRdblMmaU3yY0mCgck3cLE?usp=drive_link]() | [https://colab.research.google.com/drive/1P43cVEJcbV3bcGnBSijX84Su9YjgzTXn?usp=sharing]() |
| Ch6 | [https://colab.research.google.com/drive/11zloPjh4CEg0Saag-OX6OLVg3HPdwECF?usp=drive_link]() | [https://colab.research.google.com/drive/1ckhmwvPmijHeIwl0KdQgwgz6yxRkDTSh?usp=sharing]() |
| Ch7 | [https://colab.research.google.com/drive/11gKpZZTUuGlyOopEddLQXa9e_jls1VfM?usp=drive_link]() | [https://colab.research.google.com/drive/1-V6PNU7pmqie9poD9MbARCMPUHij7DZn?usp=sharing]() |
| Ch8 | [https://colab.research.google.com/drive/1OONiWZBb81peW5Tc-JK5yH0PlYhe33N_?usp=drive_link]() | [https://colab.research.google.com/drive/178qbFtRwRCjeJtPyR_ZYayklDcyFU56I?usp=sharing]() |
| Ch9 | [https://colab.research.google.com/drive/115DrbnoRXPU83BeerZj3uYdPU6Abm2Uu?usp=drive_link]() | [https://colab.research.google.com/drive/1k-5Ax0w4bTechrfhgrmlxV0sCLH202A-?usp=sharing]() |

## What we learned

### Chapter 1-3
We learned that raw datasets are almost always messy, incomplete, or filled with inconsistent formatting that must be cleaned before modeling. What surprised us was how missing values aren't just empty cells to be deleted—recklessly dropping them can destroy valuable context, so choosing between imputation strategies and dropping rows requires careful judgment.

### Chapter 4
We learned how machine learning models require text and categories to be converted into numerical formats like One-Hot Encoding or Label Encoding. We're surprised to realize that assigning simple numbers to categories (like 1, 2, 3) can accidentally trick a model into thinking one category is "greater than" another, which is why creating dummy binary columns is necessary for non-ordinal features.

### Chapter 5
We learned that features measured on different scales (like age vs. income) can cause models to heavily favor columns with larger numbers. What surprised us most was that standardizing a feature doesn't change its underlying distribution shape, it simply adjusts the mean to 0 and standard deviation to 1 so every variable competes fairly.

### Chapter 6
We learned how extreme values can severely skew statistical metrics and distort machine learning predictions. What surprised us was discovering that outliers aren't automatically errors to be deleted, sometimes capping them at upper/lower thresholds preserves critical data volume without throwing off the model.

### Chapter 7
We learned that more data isn't always better, as redundant or irrelevant features can actually reduce model accuracy and increase computational cost. We were surprised by how wrapper methods like RFECV test actual combinations of variables interactively, whereas filter methods look at isolated statistical relationships like correlation.

### Chapter 8
We learned how pipelines bundle cleaning, scaling, and transformation steps into a unified, reproducible sequence like a factory conveyor belt. What surprised us was how ColumnTransformer allows different feature types (like numerical vs. categorical) to pass through completely different preprocessing chains simultaneously within the same workflow.

### Chapter 9
We learned how to combine data cleaning, discretization, encoding, and visualization into an end-to-end workflow on the Titanic dataset. What surprised us was seeing how converting continuous numbers like age into discrete life-stage bins (Child, Adult, Elderly) made survival patterns immediately clear during visualization.

## Errors we found

List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

## Note on AI tools
* There are some parts of chapters we use AI about something as the description is too short for us to know and understand about it.
* 

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
