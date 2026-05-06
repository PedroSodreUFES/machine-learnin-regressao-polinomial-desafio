import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import shapiro, kstest, zscore
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1: Carregar Dados
df_receita = pd.read_csv("./dataset/sales_data.csv")
print(df_receita.describe())

# 2: Treinamento do modelo
# graus = [1,2,3,4,5,6,7,8,9,10]
graus=[2]

X = df_receita.drop(columns=['receita_em_reais'])
y = df_receita["receita_em_reais"]
colunas_numericas = ['tempo_de_experiencia', 'fator_sazonal', 'numero_de_vendas']

rmse_train_values  = []
rmse_test_values = []
percentual_rmse_values = []
r2score_test_values = []

for grau in graus:
    transformer_numericas = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocesser = ColumnTransformer(transformers=[
        ('num', transformer_numericas, colunas_numericas)
    ])

    # Criar features polinomiais
    poly_features = PolynomialFeatures(degree=grau, include_bias=False)

    model_poly = Pipeline(steps=[
        ('preprocessor', preprocesser),
        ('poly_features', poly_features),
        ('regressor', LinearRegression()),
    ])

    rmse_scores_fold_train = []
    rmse_scores_fold_test = []
    r2_scores_fold = []
    residuos_fold = []
    y_pred_total = []

    kf = KFold(n_splits=5, shuffle=True, random_state=51 )

    for train_index, test_index in kf.split(X=X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_poly.fit(X=X_train, y=y_train)

        y_train_pred = model_poly.predict(X_train)
        y_test_pred = model_poly.predict(X_test)

        rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        r2score = r2_score(y_true=y_test, y_pred=y_test_pred)
        residuos = np.array(y_test - y_test_pred) # type: ignore

        rmse_scores_fold_train.append(rmse_train)
        rmse_scores_fold_test.append(rmse_test)
        r2_scores_fold.append(r2score)
        residuos_fold.append(residuos)
        y_pred_total.append(y_test_pred)

    rmse_train_final = np.mean(rmse_scores_fold_train)
    rmse_test_final = np.mean(rmse_scores_fold_test)
    r2_final = np.mean(r2_scores_fold)
    percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
    residuos = np.array(residuos).reshape(-1)

    y_pred_total = np.array(y_pred_total).reshape(-1)
    rmse_train_values.append(rmse_train_final)
    rmse_test_values.append(rmse_test_final)
    r2score_test_values.append(r2_final)
    percentual_rmse_values.append(percentual_rmse_final)

if len(graus) > 1:
    sns.lineplot(x=graus, y=percentual_rmse_values, color='red')
    plt.ylabel("% Dif. RMSE")
    plt.xlabel("Grau")
    plt.savefig("./dataviz/polynomial_model/diff-rmse-linegraph.png")
    plt.close()

    sns.lineplot(x=graus, y=rmse_test_values, color='green', label="RMSE Test Values")
    sns.lineplot(x=graus, y=rmse_train_values, color='purple', label="RMSE Train Values")
    plt.xlabel("Grau")
    plt.savefig("./dataviz/polynomial_model/test-train-rmse-linegraph.png")
    plt.close()

    sns.lineplot(x=graus, y=r2score_test_values, color="#e6cf09")
    plt.xlabel("Graus")
    plt.savefig("./dataviz/polynomial_model/r2-scores-x-degree-linegraph.png")
    plt.close()

# Análise das Métricas
r2 = r2score_test_values[0]
rmse_test = rmse_test_values[0]
rmse_train = rmse_train_values[0]
print(f"R²-Score: {r2}")
print(f"Root Mean Squared Error - Test: {rmse_test}")
print(f"Root Mean Squared Error - Train: {rmse_train}")
# Análise Residual
residuos_std = zscore(residuos)

pg.qqplot(x=residuos_std, dist='norm', confidence=.95)
plt.savefig("./dataviz/polynomial_model/qqplot.png")
plt.close()

sns.scatterplot(x=y_test, y=residuos_std, color="purple") # type: ignore
plt.axhline(y=2, color='red')
plt.axhline(y=-2, color='red')
plt.axhline(y=0, color="#E2E3E4")
plt.savefig("./dataviz/polynomial_model/residuos-scatter.png")
plt.close()

sns.scatterplot(x=range(1, len(y_test)+1), y=y_test, color="purple", label="Y true")
sns.scatterplot(x=range(1, len(y_test)+1), y=y_test_pred, color="green", label="Y pred") # type: ignore
plt.legend()
plt.savefig("./dataviz/polynomial_model/y-true-x-y-test-scatter.png")
plt.close()

shap_stat, shap_p_value=shapiro(x=residuos_std)
ks_stat, ks_p_value = kstest(residuos_std, 'norm')
ll_stat, ll_p_value = lilliefors(x=residuos_std)
print(f"Shapiro-Wilk P-Value: {shap_p_value}")
print(f"Kolmogorov-Smirnov P-Value: {ks_p_value}")
print(f"Lilliefors P-Value: {ll_p_value}")