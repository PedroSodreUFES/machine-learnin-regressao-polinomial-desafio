import pandas as pd

# 1: Carregar Dados
df_receita = pd.read_csv("./dataset/sales_data.csv")
print(df_receita.head())

# 2: Treinamento do modelo
graus = [1,2,3,4,5,6,7,8,9,10]
# graus=[x]
