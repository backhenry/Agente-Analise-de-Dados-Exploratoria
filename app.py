import io
import zipfile
import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

# ==============================================================================
# SEÇÃO 1: CONFIGURAÇÃO INICIAL
# ==============================================================================

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Agente EDA - Análise de Dados com IA", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA)")
st.markdown(
    "Este agente usa a chave da OpenAI carregada do arquivo .env. "
    "Ele responderá a perguntas, gerará gráficos e extrairá Conclusões (via memória)."
)

if not openai_api_key:
    st.error("❌ Erro de Configuração: A chave 'OPENAI_API_KEY' não foi encontrada no ambiente.")
    st.markdown("Crie um arquivo chamado .env no diretório do script e insira: OPENAI_API_KEY='SUA_CHAVE_AQUI'")
    st.stop()

# Estado da sessão
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_visualizations" not in st.session_state:
    st.session_state.current_visualizations = {"figures": [], "dataframes": []}

# ==============================================================================
# SEÇÃO 2: UTILITÁRIOS DE CARREGAMENTO
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_extract_data(uploaded_file_bytes, filename):
    """Carrega dados de CSV ou ZIP, ajustando nomes de colunas para minúsculas."""
    if uploaded_file_bytes is None:
        return {"status": "error", "message": "Nenhum arquivo recebido."}
    bio = io.BytesIO(uploaded_file_bytes)

    try:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(bio, "r") as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_names:
                    return {"status": "error", "message": "ZIP não contém CSV."}
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(bio)
        else:
            return {"status": "error", "message": "Formato não suportado. Envie CSV ou ZIP."}

        df.columns = [c.lower() for c in df.columns]
        return {"status": "success", "df": df, "message": f"Arquivo {filename} carregado."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar arquivo: {e}"}

# ==============================================================================
# SEÇÃO 3: FERRAMENTAS DO AGENTE
# ==============================================================================

def show_descriptive_stats(*args):
    """Gera estatísticas descritivas (tipos, contagem, média, etc) para uma visão geral dos dados."""
    df = st.session_state.df
    if df is None: 
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    try:
        types_info = [f"- **{col}**: {str(df[col].dtype)} (Únicos: {df[col].nunique()}, Nulos: {df[col].isnull().sum()})" for col in df.columns]
        types_output = "**Tipos de dados e contagens:**\n" + "\n".join(types_info)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            st.session_state.current_visualizations["dataframes"].append(stats_df)

        freq_output = "\n\n**Valores mais frequentes (top 3):**\n"
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                top_values = df[col].value_counts().head(3)
                freq_output += f"\n**{col}**: " + ", ".join([f"'{k}': {v}" for k, v in top_values.items()])

        return {
            "status": "success",
            "message": types_output + freq_output
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar estatísticas: {e}"}

def generate_histogram(column: str, *args):
    """Gera histograma Plotly para coluna numérica para visualizar sua distribuição e frequência."""
    df = st.session_state.df
    col = column.lower().strip()
    if df is None or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {"status": "error", "message": f"Coluna '{col}' não encontrada ou não é numérica."}
    try:
        fig = px.histogram(df, x=col, title=f"Histograma de {col}")
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": f"Histograma gerado para '{col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar histograma: {e}"}

def generate_correlation_heatmap(*args):
    """Gera mapa de calor da correlação entre colunas numéricas para identificar relações lineares."""
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df is None or len(numeric_cols) < 2:
        return {"status": "error", "message": "Não há colunas numéricas suficientes."}
    try:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Matriz de Correlação', color_continuous_scale='RdBu_r')
        fig.update_xaxes(side='top')
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": "Mapa de calor da correlação gerado."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar mapa de calor: {e}"}

def generate_scatter_plot(columns_str: str, *args):
    """Gera scatter Plotly entre duas colunas (X, Y) para visualizar a relação entre elas (Ex: v1, v2)."""
    df = st.session_state.df
    col_names = [c for c in re.split(r"[,\s]+", columns_str.lower()) if c and c != 'e']
    if df is None or len(col_names) < 2:
        return {"status": "error", "message": "Forneça duas colunas separadas por vírgula ou espaço (Ex: v1, v2)."}
    x_col, y_col = col_names[0], col_names[1]
    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Colunas '{x_col}' ou '{y_col}' não encontradas."}

    try:
        color_col = 'class' if 'class' in df.columns else None
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": f"Gráfico de dispersão gerado para '{x_col}' vs '{y_col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar scatter plot: {e}"}

def detect_outliers_isolation_forest(*args):
    """Detecta outliers usando IsolationForest nas colunas v*, time e amount e retorna uma amostra."""
    try:
        df = st.session_state.df
        feature_cols = [c for c in df.columns if c.startswith('v')] + ['time', 'amount']
        existing = [c for c in feature_cols if c in df.columns]
        if df is None or not existing:
            return {"status": "error", "message": "Nenhuma coluna v*, time ou amount encontrada."}

        X = df[existing].dropna()
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.01, random_state=42); labels = iso.fit_predict(Xs)
        out_idx = X.index[labels == -1]; sample = df.loc[out_idx].head(10)
        msg = f"Foram detectados **{len(out_idx)}** outliers (1% do dataset) usando IsolationForest. Amostra de 10 linhas:\n\n"
        msg += sample.to_markdown(tablefmt="pipe")
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": f"Erro na detecção de outliers: {e}"}

def find_clusters_kmeans(n_clusters: str, *args):
    """Executa o agrupamento KMeans nas colunas v*, time e amount e retorna um resumo dos clusters (Ex: 3)."""
    try: n = int(n_clusters)
    except Exception: return {"status": "error", "message": "Número de clusters inválido."}

    df = st.session_state.df
    feature_cols = [c for c in df.columns if c.startswith('v')] + ['time', 'amount']
    existing = [c for c in feature_cols if c in df.columns]
    if df is None or not existing:
        return {"status": "error", "message": "Nenhuma coluna v*, time ou amount encontrada."}

    try:
        X = df[existing].fillna(0)
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=n, random_state=42, n_init=10); labels = km.fit_predict(Xs)
        df_copy = df.copy(); df_copy['cluster'] = labels
        summary = df_copy.groupby('cluster')[existing].mean().to_markdown(tablefmt='pipe')
        return {"status": "success", "message": f"KMeans executado com {n} clusters:\n\n{summary}"}
    except Exception as e:
        return {"status": "error", "message": f"Erro no K-Means: {e}"}

tool_functions = [
    show_descriptive_stats,
    generate_histogram,
    generate_correlation_heatmap,
    generate_scatter_plot,
    detect_outliers_isolation_forest,
    find_clusters_kmeans,
]

# ==============================================================================
# SEÇÃO 4: INICIALIZAÇÃO DO AGENTE
# ==============================================================================

def initialize_agent(api_key):
    """Configura o LLM, Prompt, Memória e AgentExecutor."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.0)
    memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)
    
    system_prompt = (
        "Você é um agente de EDA especializado em análise de dados financeiros e de fraude. "
        "SEMPRE use as ferramentas disponíveis para gerar cálculos, análises e gráficos. "
        "Responda em Português com insights claros. "
        "As colunas estão em minúsculas: v1-v28, time, amount, class. "
        "IMPORTANTE: Quando uma ferramenta gerar um gráfico ou tabela, confirme isso na sua resposta. "
        "REQUISITO DE MEMÓRIA: Quando solicitado conclusões, sintetize TODAS as análises feitas."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools_as_langchain = [Tool(name=fn.__name__, description=fn.__doc__, func=fn) for fn in tool_functions]
    agent = create_tool_calling_agent(llm, tools_as_langchain, prompt)

    return AgentExecutor(
        agent=agent, tools=tools_as_langchain, verbose=True, memory=memory,
        max_iterations=15, handle_parsing_errors=True
    )

# ==============================================================================
# SEÇÃO 5: INTERFACE E LOOP DE CHAT
# ==============================================================================

with st.sidebar:
    st.header("📁 Upload de dados")
    uploaded_file = st.file_uploader("CSV ou ZIP (com CSV)", type=["csv", "zip"])

    if st.button("🚀 Carregar e Inicializar Agente") and uploaded_file is not None:
        st.session_state.df = None
        st.session_state.messages = []
        st.session_state.agent_executor = None
        st.session_state.current_visualizations = {"figures": [], "dataframes": []}

        with st.spinner("Carregando arquivo e preparando o agente..."):
            data = load_and_extract_data(uploaded_file.getvalue(), uploaded_file.name)

        if data['status'] == 'success':
            st.session_state.df = data['df']
            try:
                st.session_state.agent_executor = initialize_agent(openai_api_key)
                st.success("✅ Agente inicializado! DataFrame carregado.")
            except Exception as e:
                st.error(f"Erro ao inicializar agente: {e}")
        else:
            st.error(data['message'])

    if st.session_state.df is not None:
        st.success(f"📊 DataFrame: {len(st.session_state.df)} linhas, {len(st.session_state.df.columns)} colunas")
        if st.checkbox("👀 Ver amostra dos dados"):
            st.dataframe(st.session_state.df.head())

# Exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg["content"])
        if "figures" in msg and msg["figures"]:
            for fig in msg["figures"]:
                st.plotly_chart(fig, use_container_width=True)
        if "dataframes" in msg and msg["dataframes"]:
            for df_viz in msg["dataframes"]:
                st.dataframe(df_viz, use_container_width=True)

# Loop de chat
if prompt := st.chat_input("Pergunte algo sobre os dados..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.agent_executor is None:
        st.error("❌ Agente não inicializado.")
    else:
        assistant_placeholder = st.chat_message("assistant")
        
        with assistant_placeholder:
            with st.spinner("🤖 Analisando..."):
                try:
                    st.session_state.current_visualizations = {"figures": [], "dataframes": []}
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    
                    figures = st.session_state.current_visualizations["figures"].copy()
                    dataframes = st.session_state.current_visualizations["dataframes"].copy()
                    
                    response_text = response.get('output', 'Erro ao processar.')
                    st.markdown(response_text)
                    
                    for fig in figures:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    for df_viz in dataframes:
                        st.dataframe(df_viz, use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "figures": figures,
                        "dataframes": dataframes
                    })

                except Exception as e:
                    error_msg = f"❌ Erro: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.expander("💡 Exemplos de perguntas"):
    st.markdown("""
    1. Quais são os tipos de dados e as estatísticas descritivas?
    2. Detecte outliers nos dados.
    3. Gere o mapa de calor das correlações.
    4. Crie um gráfico de dispersão entre time e amount.
    5. Qual a conclusão geral?
    """)
