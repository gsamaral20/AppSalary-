# %%
# # %% Simula a mecânica do jupyter notebook

import pandas as pd

df = pd.read_csv("data/dataset.csv")
df.head()

# %%

features = {
    "1.a_idade": "idade", 
    "1.b_genero": "genero", 
    "1.d_pcd": "pcd", 
    "1.i.1_uf_onde_mora": "ufOndeMora", 
    "2.f_cargo_atual": "cargoAtual", 
    "2.g_nivel": "nivel", 
    "2.i_tempo_de_experiencia_em_dados": "tempoDeExperienciaEmDados", 
    "2.j_tempo_de_experiencia_em_ti": "tempoDeExperienciaEmTi", 
    #"2.l.4_Flexibilidade de trabalho remoto": "flexibilidadeDeTrabalhoRemoto"
}

# seleciona todas as colunas, shift alt i e ai replica em todas as linhas
# selecione o nome exemplo Flexibilidade de trabalho remoto, no topo em procurar >camel case e ai edita automaticamente

target = "2.h_faixa_salarial"

columns = list(features.keys()) + [target]

df = df[columns].copy()

df.rename(columns=features, inplace=True)


df.head()


# %%
df[target].unique()

# Padronizando os valores da coluna target
depara_salario = {
    'de R$ 1.001/mês a R$ 2.000/mês': '01 - de R$ 1.001/mês a R$ 2.000/mês',
    'Menos de R$ 1.000/mês': '03 - Menos de R$ 1.000/mês', 
    'de R$ 6.001/mês a R$ 8.000/mês': '04 - de R$ 6.001/mês a R$ 8.000/mês', 
    'de R$ 3.001/mês a R$ 4.000/mês': '05 - de R$ 3.001/mês a R$ 4.000/mês',
    'de R$ 2.001/mês a R$ 3.000/mês': '06 - de R$ 2.001/mês a R$ 3.000/mês', 
    'de R$ 4.001/mês a R$ 6.000/mês': '07 - de R$ 4.001/mês a R$ 6.000/mês',
    'de R$ 8.001/mês a R$ 12.000/mês': '08 - de R$ 8.001/mês a R$ 12.000/mês',
    'de R$ 12.001/mês a R$ 16.000/mês': '09 - de R$ 12.001/mês a R$ 16.000/mês',
    'de R$ 30.001/mês a R$ 40.000/mês': '10 - de R$ 30.001/mês a R$ 40.000/mês',
    'de R$ 20.001/mês a R$ 25.000/mês': '11 - de R$ 20.001/mês a R$ 25.000/mês',
    'de R$ 16.001/mês a R$ 20.000/mês': '12 - de R$ 16.001/mês a R$ 20.000/mês',
    'de R$ 25.001/mês a R$ 30.000/mês': '13 - de R$ 25.001/mês a R$ 30.000/mês', 
    'Acima de R$ 40.001/mês': '14 - Acima de R$ 40.001/mês'
}

df[target] = df[target].replace(depara_salario)

df[target].isna().sum()

df_not_na = df[~df[target].isna()]

df_not_na[target].isna().sum()

# %%
X = df_not_na[features.values()]
y = df_not_na[target]

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=42,
                                                                    stratify=y)
X_test.shape

# %%
# Verificar o % de NA presente nas colunas do df, por exemplo Flexibilidade de trabalho remoto tem 68% de dados NA
X_train.isna().mean()

X_train.to_csv("data/template.csv", index=False)

# Optei por comentar no começo o Flexibilidade
# %%
from feature_engine import imputation
from feature_engine import encoding

input_classe = imputation.CategoricalImputer(fill_value="Não informado",
                                             variables=["ufOndeMora", "cargoAtual", "nivel"])

X_train.dtypes

# Não é bom usar get_dummies porque ao aplicar na base de treino e teste não há garantias de que as mesmas categorias terão a mesma classificação
onehot = encoding.OneHotEncoder(variables= [
    'genero',                   
    'pcd',                      
    'ufOndeMora',               
    'cargoAtual',               
    'nivel',                    
    'tempoDeExperienciaEmDados',
    'tempoDeExperienciaEmTi',
])

# Outra opção é encoding.MeanEncoder
# %%
from sklearn import ensemble
from sklearn import pipeline

clf = ensemble.RandomForestClassifier(random_state=42, 
                                      min_samples_leaf=20)
# Os steps que o dado vai passar
modelo = pipeline.Pipeline(
    steps=[('imputador', input_classe),
           ('enconder', onehot),
           ('algoritmo', clf)]
)

# %% 
# mlflow faz a gestão do ciclo de vida do modelo
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=876263473380248139)


# %%
from sklearn import metrics

with mlflow.start_run():
    mlflow.sklearn.autolog()

    modelo.fit(X_train, y_train)
    y_train_predict = modelo.predict(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    y_teste_predict = modelo.predict(X_test)
    acc_test = metrics.accuracy_score(y_test, y_teste_predict)


# %%
