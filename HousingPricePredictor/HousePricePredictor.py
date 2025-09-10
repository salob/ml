import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

tracker = EmissionsTracker()
tracker.start()

housing=fetch_california_housing()
data=pd.DataFrame(housing['data'],columns=housing['feature_names'])
df=data.copy()
df['MedHouseVal']=housing['target']
df.head()
df.shape
df.size
df.columns
df.info()
df.isnull().sum()
df.describe(include='all')
corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Matrix',fontsize=18)
plt.xlabel('Variables',fontsize=14)
plt.ylabel('Variables',fontsize=14)
plt.show()

tracker.stop()