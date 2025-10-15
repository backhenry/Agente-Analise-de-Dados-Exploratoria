# /app.py

import os
import streamlit as st
from dotenv import load_dotenv
import ui

# Carrega as variáveis de ambiente (sua chave da OpenAI)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Configuração da Página ---
st.set_page_config(page_title="Agente EDA - Análise de Dados com IA", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA)")

# Verifica se a chave da API está configurada
if not openai_api_key:
    st.error("❌ Chave 'OPENAI_API_KEY' não encontrada.")
    st.markdown("Crie um arquivo `.env` e adicione: `OPENAI_API_KEY='SUA_CHAVE_AQUI'`")
    st.stop()

# --- Inicialização do Estado da Sessão ---
# Apenas o mínimo necessário aqui. O resto é gerenciado nos outros módulos.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_visualizations" not in st.session_state:
    st.session_state.current_visualizations = {"figures": [], "dataframes": []}

# --- Execução da Interface ---
ui.setup_sidebar(openai_api_key)
ui.render_chat_history()
ui.handle_chat_input()

# Adiciona o expander com exemplos no final
with st.expander("💡 Ideias de Prompts"):
    st.markdown("""

    1.  **Visão Geral:**
        * `Para começar, me dê um resumo completo e detalhado dos dados.`

    2.  **Limpeza de Dados:**
        * `Notei que a coluna 'age' tem valores ausentes. Por favor, preencha esses valores usando a mediana.`

    3.  **Análise Univariada:**
        * `Entendido. Agora, mostre-me a distribuição de passageiros por classe com um gráfico de barras.`

    4.  **Insight Principal (Agregação + Gráfico):**
        * `Qual era a taxa de sobrevivência por sexo? Mostre os dados e um gráfico de barras.`

    5.  **Análise Avançada (Tabela Dinâmica):**
        * `Vamos mais fundo. Crie uma tabela dinâmica que mostre a taxa de sobrevivência para cada sexo DENTRO de cada classe.`

    6.  **Teste de Hipótese (Estatística):**
        * `Teste se a tarifa média ('fare') dos sobreviventes (grupo 1) é estatisticamente diferente da dos não sobreviventes (grupo 0).`

    7.  **Engenharia de Features:**
        * `Vamos criar uma nova métrica. Crie uma coluna 'family_size' somando 'sibsp', 'parch' e 1.`
    
    8.  **Análise de Texto:**
        * `Para finalizar, gere uma nuvem de palavras da coluna 'name' para ver os termos mais comuns.`
    """)