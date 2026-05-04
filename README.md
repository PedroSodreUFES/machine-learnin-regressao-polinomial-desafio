![Python](https://img.shields.io/badge/Python-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-FF6F00?style=for-the-badge&logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-FF6F00?style=for-the-badge&logo=fastapi&logoColor=white)
![Sreamlit](https://img.shields.io/badge/Streamlit-FF6F00?style=for-the-badge&logo=streamlit&logoColor=white)
![Scipy](https://img.shields.io/badge/SciPy-FF6F00?style=for-the-badge&logo=scipy&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-FF6F00?style=for-the-badge&logo=python&logoColor=white)
![Pingouin](https://img.shields.io/badge/Pingouin-FF6F00?style=for-the-badge&logo=python&logoColor=white)
# Modelo: Previsibilidade de Salário
Um modelo em **Regressão Polinomial** para prever qual o salário de uma pessoa de uma empresa fictícia de acordo com seus dados, sendo estes:
+ Nível na empresa (1-10)
+ Tempo na empresa (em meses)
## Sobre o projeto
1. Trata de uma análise exploratória de dados para verificar a relação dos dados com a variável target salario. Feita com pandas, seaborn e matplotlib.
2. Com o pairplot, é possível notar a relação entre as variáveis. Além disso, com o heatmap das correlações é possível ver quais variáeveis independentes tem mais realação de pearson e spearman com o salário.
3. O treinamento do modelo é feito com vários graus polinomiais distintos até encontras um ótimo. Após encontrar um ótimo, realiza o treinamento a partir somente do ótimo. Também é feito o treinamento de um modelo linear para fins de comparação e mostrar como o modelo não é bom para esse cenário.
4. Após o treinamento do modelo, há uma análise da qualidade do modelo, usando métricas como erro médio absoluto, erro médio na raíz quadrada e r2_score.
5. Faz-se uma análise dos resíduos da solução, olhando seu testes de normalidade e de homocedasticidade para ver se estão próximos a uma distribuição normal.
6. Usa joblib para salvar o modelo para consumo em um arquivo .pkl. ESse consumo é feito pelo App criado com Streamlit e por uma api feita com FastAPI.
## Tecnologias usadas
1. Python
2. Scikit-Learn
3. Seaborn
4. Matplotlib
5. Pandas
6. Scipy
7. FastAPI
8. Streamlit
9. Joblib
10. Pingouin
### Como preparar o ambiente
```bash
pipenv sync
pipenv shell
```
### Como testar em forma de api
```bash
uvicorn api_modelo_salario:app --reload
```
Ir na web no [link de teste](http://localhost:8000/docs)
### Como testar em forma de web app
Tenha a API rodando em algum terminal.
```bash
streamlit run app_modelo_salario.py
```
Ir na web no [link de teste](http://localhost:8501)
### Como rodar o código que gera o modelo
```bash
python model.py
```
## Aspectos do Modelo Treinado
### Análise do cenário

![Pairplot Dados](./dataviz/pairplot.png)

#### Variáveis numéricas
1. Pelo pairplot é possível enxergar que em termos das variáveis numericas, o nível não afeta o salário de forma clara. Enquanto há uma correlação claramente positiva entre salário e tempo na empresa.
2. Pelos histogramas na diagonal, vê-se que o tempo de empresa e o nível aparecem em uma distribuição uniforme. Contudo, o salário aparece com assimetria positiva, ou seja: muitos ganham pouco e poucos ganham muito. Isso revela que a média não é um valor interessante paras representar dados faltantes, caso tenha.
3. Com os boxplots, vê-se que não há dados faltantes em nenhuma coluna.
#### Variávies categóricas
Não há variáveis categóricas.

### Correção dos dados e salário
#### Correlação de Spearman
![Correlacao com salário spearman](./dataviz/correlacoes-spearman-heatmap.png)

#### Correlação de Pearson
![Correlacao com salário spearman](./dataviz/correlacoes-pearson-heatmap.png)

Pela correlação de Spearman é possível ver uma correlação positiva perfeita entre salário e tempo na empresa. O mesmo não pode ser visto na correlação de Pearson. Isso revela que um modelo linear não é ideal para representar o cenário, já que a correlação de Pearson é mais precisa para modelos lineares, enquanto a de Spearman enxerga relações não-lineares.

### Treinamento do modelo
![Percentual Dif. RMSE y_pred_test e y_pred_train](./dataviz/dif-percentage-rmse-por-grau-polinomial.png)

![Dif. RMSE y_pred_test e y_pred_train](./dataviz/rmse-por-grau-polinomial.png)

Usou-se o método k-fold que divide o modelo em 5, usando embaralhamento dos dados. Com isso realiza testes repetidamente com 4 dados de treinamento 1 dado de teste. Com essa iteração tira os valores R²-Score, Rounded Mean Squared Error para os dados de treinamento e de teste, e obtém os resíduos. Isso é feito tanto para o modelo linear como para o modelo polinomial.<br />
Contudo, para o modelo polinomial, isso é feito para 10 graus polinomiais com o objetivo de compreender qual polinômio representa melhor o modelo. Ao final o **grau 4** aparentou ser o ótimo por apresentar uma diferença percentual de rmse entre predições de treino e de testes menores. Além disso, para esses graus, seu RMSE são muito parecidos para teste e treinamento, o que mostra que está conseguindo lidar com ambos os cenários sem mostrar underfitting e overfitting.

### Comparação modelo linear múltiplo x modelo polinomial
|Modelo\Métricas| R²-Score | RMSE Teste | RMSE treinamento |
|:---:|:---:|:---:|:---:|
|Regressão Linear Múltipla|≃ 0.83|≃ R$ 1886.45|≃ R$ 1882.45|
|Regressão Polinomial|≃ 0.997|≃ R$ 54.71|≃ R$ 53.33|

1. ***R²-Score***: Mostra que o modelo polinomial explica melhor a variabilidade dos dados.
2. ***RSME***: Mostra que o modelo linear pode errar um salário por até 1882 reais para dados de treinamento, e até 1886 reais para dados de teste. Isso é mais que um salário mínimo, o que é muito ruim. Já o modelo polinomial erra em torno de 50 reais, o que é bem menos, algo em torno de 3.5% de um salário mínimo em Maio de 2026 

> ***Conclusão***: O modelo polinomial explica melhor a variabilidade do dataset e é mais adequado.

### Métricas do modelo
#### Métricas de linearidade e de outliers
![Linearidade](./dataviz/rgpoly-residuos-scatter.png)

1. Outliers: Pelos scatter dos resíduos, vê-se alguns pontos acima de +2 e abaixo de -2, logo há alguns outliers que não são bem explicados pelo modelo.
2. Modelo polinomial adequado e homocedasticidade: Os resíduos estão espalhados sem formar um padrão, o que indica que o modelo polinomial é adequado e há homocedasticidade.

#### Métricas de Normalidade dos Resíduos
![qqplot modelo polinomial](./dataviz/rgpoly-qqplot.png)

- QQplot indica evidência de normalidade dos resíduosjá que os resíduos não escapam do limite traçado.

| P-valor de Shapiro-Wilk | P-valor de Kolmogorov-Smirnov | P-valor de Lilliefors |
|:--:|:--:|:--:|
|≃ 0.68|≃ 1*10⁻²¹⁴|≃ 0.96|

> **H0**: *os resíduos seguem uma distribuição normal*<br/>
> **H1**: *os resíduos não seguem uma distribuição normal*
- Por ser abaixo de 0.05, *Kolmogorov-Smirnov* rejeita a hipótese nula por haver evidência de distribuição não normal nos resíduos. OBS: Kolmogorov-Smirnov rejeitam fortemente.
- Por ser acima de 0.05, *Lilliefors* e *Shapiro-Wilk* apontam evidência de distribuição normal dos resíduos, logo não há evidência suficiente para rejeitar a hipótese nula. OBS: Lilliefors não rejeita fortemente.

> *OBS: As estatísticas são mais sensíveis a outliers que o QQPlot. Por isso, o último é levado em questão para afirmar que há distribuição normal com suorte nos P-Valores de Shapiro-Wilk e Lilliefors.*


### Conclusão
  - Pela correlação de Spearman e pela correlação de Pearson já seria possível concluir que o modelo de regressão linear não é ideal para o caso devido à relação polinomial com grau acima de 1 entre tempo de empresa e salario. Isso era possível pois a correlação de Pearson indica correlação 0.9, já a de spearman indica 1. Ou seja, a correlação é mais adequada para um modelo polinomial. No entanto, com as métricas mostradas acima, mostrou-se que o modelo polinomial tem erros menores e conseguem prever melhor tanto os dados treinados quanto os de teste, além de prover um R²-Score maior que o modelo linear, ou seja explica melhor o dataset que o modelo linear.
- Como o modelo tem um R²-Score de 0.997 e tem um desvio de erro de em média 50 reais tanto para os dados de teste, quanto para os dados treinados, o modelo é adequado para prever o dataset.

### Créditos
Pedro Malini, 3 de Maio de 2026 