# %% [markdown]
# ## Linear Regression 

# %% [markdown]
# _Mise en place_ :
# - train and test the model to predict a new outcome og future employees
# - test the quality of the model. 

# %% [markdown]
# ### imports
# 

# %%
# for data storage and manipulation
import pandas
import numpy
# for diagrams and plotting
import matplotlib.pyplot as plt
import seaborn as sns
# for ML/machine learning methods and algorithms
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
# for serialization and deserialization of data from and to a file
import pickle
# for normalization of data
from sklearn.preprocessing import MinMaxScaler


# %% [markdown]
# ### load the data 

# %%
DFemployeeData = pandas.read_csv('../data/Emp-Attrition-Initial-Clean.csv')

# %%
DFemployeeData.shape

# %%
DFemployeeData.head(5)

# %%
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(DFemployeeData)
# colimns = DFemployeeData.columns makes sure that its the original column names
normalizedEmployeeData = pandas.DataFrame(scaler.fit_transform(DFemployeeData), columns=DFemployeeData.columns)

# %% [markdown]
# ### normalize the data 

# %%
normalizedEmployeeData.head(5)

# %%
list(DFemployeeData)

# %%
hist = DFemployeeData[['YearsAtCompany', 'MonthlyIncome']].hist()

# %%
# to get a overview of how many are in each category
sns.distplot(DFemployeeData['YearsAtCompany'],  label='MonthlyIncome', norm_hist=True) 

# %%
DFemployeeData[["YearsAtCompany", "MonthlyIncome"]].describe()

# %% [markdown]
# #### checking the outliers

# %%

# using the normalized data here. because the income is such a high number in relation to the years at the company
boxplot = normalizedEmployeeData.boxplot(column=['YearsAtCompany', 'MonthlyIncome']) 

# %% [markdown]
# #### Removing the outliers. 
# ##### its makes our future model more precise. because we want our predictions to be based on the most data and not the few that have a very high or very low salary

# %%
upper_limitInc = DFemployeeData["MonthlyIncome"].quantile(0.95) 
lower_limitInc = DFemployeeData["MonthlyIncome"].quantile(0.05)
upper_limitYear = DFemployeeData["YearsAtCompany"].quantile(0.95) 
lower_limitYear = DFemployeeData["YearsAtCompany"].quantile(0.05)


no_outliers = DFemployeeData[(DFemployeeData["MonthlyIncome"] < upper_limitInc) & (DFemployeeData["MonthlyIncome"] > lower_limitInc) ]
no_outliers = no_outliers[(no_outliers["YearsAtCompany"] < upper_limitYear) & (no_outliers["YearsAtCompany"] > lower_limitYear) ]

no_outliers

# %%
outliers = pandas.concat([DFemployeeData, no_outliers]).drop_duplicates(keep=False)
outliers



# %%
DFemployeeDataWithOutliers = no_outliers[["MonthlyIncome", "YearsAtCompany"]].corr()

# %% [markdown]
# ### heat map with correlation between MonthlyIncome and YearsAtCompany. with no outliers

# %%
plt.figure(figsize=(16,8)) # makes the diagram bigger.
sns.heatmap(DFemployeeDataWithOutliers, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between MothlyIncome and yearsAtCompany without MothlyIncome outliers')
plt.show()

# %% [markdown]
# ### conclusion there is no correlation in the MonthlyIncome and YearsatCompany woth no outliers

# %%
DFemployeeDataCorrelation = DFemployeeData[["MonthlyIncome", "YearsAtCompany"]].corr()

# %%
plt.figure(figsize=(16,8)) # makes the diagram bigger.
sns.heatmap(DFemployeeDataCorrelation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between MothlyIncome and yearsAtCompany')
plt.show()

# %% [markdown]
# ### Conclusion: with all the outliers with correlation is moderate positive. but not very precise to predict 

# %%
# Select columns that might influence MonthlyIncome
cols_to_check = [
    "MonthlyIncome",
    "Age",
    "Education",
    "JobLevel",
    "TotalWorkingYears",
    "YearsAtCompany",
    "PerformanceRating",
    "PercentSalaryHike",
    "NumCompaniesWorked",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# Create a new DataFrame with only those columns
DFemployeeDataCheckCorr = DFemployeeData[cols_to_check].corr()

# %%
plt.figure(figsize=(16,8)) # makes the diagram bigger.
sns.heatmap(DFemployeeDataCheckCorr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between MothlyIncome and alot of other columns')
plt.show()

# %% [markdown]
# ### Conclusion "JobLevel" and "TotalWorkingYears" have the most impact on Monthlyincome

# %%
# Features and target. X = feature, y = target
X = DFemployeeData[["TotalWorkingYears"]]  # Removed JobLevel to get a better look at years and income
y = DFemployeeData["MonthlyIncome"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)



# %%
#  Predict and Validate
y_pred = model.predict(X_test)


# %%
print("R² score:", sm.r2_score(y_test, y_pred))
print("MAE:", sm.mean_absolute_error(y_test, y_pred))

# %% [markdown]
# ### with both "Joblevet" and "TotalWorkingYears"
# ### R^2 score 0.89 = The model explains 89% of the variation in MonthlyIncome. this is good
# ### MAE 1169 = on average the income can be 1170 $ lower or higher from the true income

# %%
a = model.coef_ # the slope of the line = how much monthly income increases with each unit from the features/independent variables(Joblevel and TotalWorkingYears)
b = model.intercept_ # the y intercept

# %%
#Visualize the linear regression
plt.scatter(X["TotalWorkingYears"], y, cmap='viridis')
plt.colorbar(label='MonthlyIncome')
plt.xlabel('TotalWorkingYears')
plt.ylabel('MonthlyIncome')
plt.title('TotalWorkingYears colored by MonthlyIncome')
plt.plot(X_train, a*X_train + b, color='blue')
plt.plot(X_test, y_pred, color='orange')
plt.show()



# %% [markdown]
# ### conclusion : The regression line (orange) fits the data quite well, showing that the model can predict income based on experience.

# %% [markdown]
# ### conclusion : to get more precise prediction. we should make a Multiple linear regression with "JobLevel" and "TotalWorkingYears" as the independent variables. since the R^2 model would have been 0.89 instead of 0.59 (with only "TotalWorkingYears") 
# 

# %% [markdown]
# 


