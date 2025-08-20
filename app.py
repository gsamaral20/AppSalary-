# %% 
import pandas as pd
import streamlit as st
import mlflow

# Função cacheada 1 dia, assim não precisa ficar carregando toda vez
@st.cache_resource(ttl='1day')
def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # O /1 é a versão do modelo, eu posso alterar o modelo em train.py, enviar para o mlflow, registrar esse novo modelo em salario-model e ai altero a versão
    # model = mlflow.sklearn.load_model("models:///salario-model/1")

    models = [i for i in mlflow.search_registered_models() if i.name == "salario-model"] 
    last_version = max([int(i.version) for i in models[0].latest_versions])

    model = mlflow.sklearn.load_model(f"models:///salario-model/{last_version}")

    return model

model = load_model()

# %% 
data_template = pd.read_csv("data/template.csv")

st.dataframe(data_template.head())

# %%
st.markdown("# Data Salary")

col1, col2, col3 = st.columns(3)
with col1: 
    idade = st.number_input("Idade", min_value=data_template['idade'].min(),
                                    max_value=100)
    genero = st.selectbox("Gênero", options=data_template['genero'].unique())
    pcd = st.selectbox("PCD", options=data_template['pcd'].unique())
    ufs = data_template['ufOndeMora'].sort_values().unique().tolist()
    uf = st.selectbox("UF", options=ufs)
with col2:
    cargos = data_template['cargoAtual'].sort_values().unique().tolist()
    cargo = st.selectbox("Cargo Atual", options=cargos)
    niveis = data_template['nivel'].sort_values().unique().tolist()
    nivel = st.selectbox("Nível", options=niveis)
with col3:
    temp_dados = data_template['tempoDeExperienciaEmDados'].sort_values().unique().tolist()
    temp_exp_dados = st.selectbox("Tempo de Experiência em Dados", options=temp_dados)
    temp_it = data_template['tempoDeExperienciaEmTi'].sort_values().unique().tolist()
    temp_exp_it = st.selectbox("Tempo de Experiência em IT", options=temp_dados)


ok = st.button("Calcular Salário")
if ok:
    data = pd.DataFrame([{
        'idade': idade,
        'genero': genero,                   
        'pcd': pcd,                      
        'ufOndeMora': uf,               
        'cargoAtual': cargo,               
        'nivel': nivel,                    
        'tempoDeExperienciaEmDados': temp_exp_dados,
        'tempoDeExperienciaEmTi': temp_exp_it,
    }])
    
    salario = model.predict(data[model.feature_names_in_])[0]
    salario = salario.split("- ")[-1]

    st.markdown(f"Sua faixa salarial: `{salario}`")
    # st.dataframe(data)

# %%

# salarios_posicao = {
#     "junior": 4000,
#     "pleno": 7500,
#     "senior": 11000
# }

# col1, col2 = st.columns(2)

# with col1: 
#     select_box_posicao = st.selectbox("Senioridade", options = salarios_posicao.keys())
# with col2: 
#     input_tempo_experiencia = st.number_input("Tempo de experiência", min_value=0, max_value=35, help="Seu tempo de mercado em anos")

# salario = salarios_posicao[select_box_posicao] + input_tempo_experiencia * 500

# st.markdown(f"Seu salário é: R$ {salario:.2f}")
