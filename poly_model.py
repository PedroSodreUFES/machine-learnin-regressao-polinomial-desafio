import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1: Carregar Dados
df_receita = pd.read_csv("./dataset/sales_data.csv")
print(df_receita.head())

# 2: Treinamento do modelo
graus = [1,2,3,4,5,6,7,8,9,10]
# graus=[8]

for grau in graus:

