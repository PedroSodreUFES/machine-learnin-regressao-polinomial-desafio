import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import shapiro, kstest, zscore
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1: Carregar dados
print("Carregar Dados")
df_vendas = pd.read_csv("./dataset/sales_data.csv")
print(df_vendas.info())

# 2: Análise Exploratória dos dados
print("\nAnálise Exploratória:")
sns.heatmap(
    data=df_vendas
        .corr("spearman")[['receita_em_reais']]
        .sort_values(by="receita_em_reais", ascending=False),
    vmax=1,
    vmin=-1,
    cmap='crest',
    annot=True,
)
plt.savefig("./dataviz/receita-spearman-heatmap.png")
plt.close()

sns.heatmap(
    data=df_vendas
        .corr("pearson")[['receita_em_reais']]
        .sort_values(by="receita_em_reais", ascending=False),
    vmax=1,
    vmin=-1,
    cmap='crest',
    annot=True,
)
plt.savefig("./dataviz/receita-pearson-heatmap.png")
plt.close()

sns.pairplot(data=df_vendas)
plt.savefig("./dataviz/pairplot.png")
plt.close()

sns.scatterplot(data=df_vendas, x='numero_de_vendas', y='receita_em_reais')
plt.xlabel("Número de vendas")
plt.ylabel("Receita em BRL")
plt.savefig("./dataviz/vendas-x-receitas.png")
plt.close()

sns.scatterplot(data=df_vendas, x='fator_sazonal', y='receita_em_reais')
plt.xlabel("Fator sazonal")
plt.ylabel("Receita em BRL")
plt.savefig("./dataviz/fator-sazonal-x-receitas.png")
plt.close()

sns.scatterplot(data=df_vendas, x='tempo_de_experiencia', y='receita_em_reais')
plt.xlabel("Tempo de Experiência")
plt.ylabel("Receita em BRL")
plt.savefig("./dataviz/tempo-experiencia-x-receitas.png")
plt.close()

sns.boxplot(data=df_vendas, x='tempo_de_experiencia')
plt.xlabel("Tempo de Experiência")
plt.savefig("./dataviz/tempo-experiencia-boxplot.png")
plt.close()

sns.boxplot(data=df_vendas, x='fator_sazonal')
plt.xlabel("Fator Sazonal")
plt.savefig("./dataviz/fator-sazonal-boxplot.png")
plt.close()

sns.boxplot(data=df_vendas, x='numero_de_vendas')
plt.xlabel("Número de Vendas")
plt.savefig("./dataviz/numero-de-vendas-boxplot.png")
plt.close()

sns.boxplot(data=df_vendas, x='receita_em_reais')
plt.xlabel("Receita em Reais")
plt.savefig("./dataviz/receita-boxplot.png")
plt.close()

# 3: Treinar modelo
print("\nTreinar Modelo")
X = df_vendas.drop(columns=['receita_em_reais'])
y = df_vendas["receita_em_reais"]

kf = KFold(n_splits=5, shuffle=True, random_state=51 )

colunas_numericas = ['tempo_de_experiencia', 'fator_sazonal', 'numero_de_vendas']

transformer_numericas = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocesser = ColumnTransformer(transformers=[
    ('num', transformer_numericas, colunas_numericas)
])

model_linear = Pipeline(steps=[
    ("preprocessor", preprocesser),
    ("regressor", LinearRegression())
])

# Aplicar K-Folds
rmse_scores_fold_train = []
rmse_scores_fold_test = []
r2_scores_fold = []
residuos_fold = []
y_pred_total = []

for train_index, test_index in kf.split(X=X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_linear.fit(X=X_train, y=y_train)

    y_train_pred = model_linear.predict(X_train)
    y_test_pred = model_linear.predict(X_test)

    rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    r2score = r2_score(y_true=y_test, y_pred=y_test_pred)
    residuos = np.array(y_test - y_test_pred) # type: ignore

    rmse_scores_fold_train.append(rmse_train)
    rmse_scores_fold_test.append(rmse_test)
    r2_scores_fold.append(r2score)
    residuos_fold.append(residuos)
    y_pred_total.append(y_test_pred)

r2score_final = np.mean(r2_scores_fold)
rmse_train_final = np.mean(rmse_scores_fold_train)
rmse_test_final = np.mean(rmse_scores_fold_test)
y_pred = np.array(y_pred_total).reshape(-1)
residuos_final = np.array(residuos).reshape(-1)

print("\nMétricas:")
print(f"R²-Score final: {r2score_final}")
print(f"Root Mean Squared Error - Train: {rmse_train_final}")
print(f"Root Mean Squared Error - Test: {rmse_test_final}")
