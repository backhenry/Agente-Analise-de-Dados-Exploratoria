# /ui.py

import streamlit as st
from data_loader import load_and_process_data
from agent import initialize_agent

def setup_sidebar(api_key):
    """Configura a barra lateral para upload de arquivos e inicialização do agente."""
    with st.sidebar:
        st.header("📁 Upload de dados")
        uploaded_file = st.file_uploader("CSV ou ZIP (contendo CSV)", type=["csv", "zip"])

        if st.button("🚀 Carregar e Inicializar Agente") and uploaded_file is not None:
            st.session_state.clear()
            st.session_state.messages = []

            data = load_and_process_data(uploaded_file)

            if data['status'] == 'success':
                st.session_state.df = data['df']
                try:
                    st.session_state.agent_executor = initialize_agent(api_key)
                    st.success("✅ Agente pronto! DataFrame carregado.")
                except Exception as e:
                    st.error(f"Erro ao inicializar agente: {e}")
            else:
                st.error(data['message'])

        if "df" in st.session_state and st.session_state.df is not None:
            st.success(f"📊 DataFrame: {len(st.session_state.df)} linhas, {len(st.session_state.df.columns)} colunas")
            if st.checkbox("👀 Ver amostra dos dados"):
                st.dataframe(st.session_state.df.head())

def render_chat_history():
    """Exibe o histórico de mensagens e visualizações no chat."""
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg['role']):
            st.markdown(msg["content"])
            
            # Exibe todas as visualizações salvas no histórico
            for fig in msg.get("figures", []):
                st.plotly_chart(fig, use_container_width=True)
            for df_viz in msg.get("dataframes", []):
                st.dataframe(df_viz, use_container_width=True)
            for fig_mpl in msg.get("matplotlib_figs", []):
                st.pyplot(fig_mpl)

def handle_chat_input():
    """Gerencia a entrada do usuário e a execução do agente."""
    if prompt := st.chat_input("Pergunte algo sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "agent_executor" not in st.session_state or st.session_state.agent_executor is None:
            st.error("❌ Agente não inicializado. Carregue um arquivo e clique em 'Inicializar'.")
            return

        with st.chat_message("assistant"):
            with st.spinner("🤖 Analisando..."):
                try:
                    # Inicializa o container de visualizações para a rodada atual
                    st.session_state.current_visualizations = {
                        "figures": [], 
                        "dataframes": [],
                        "matplotlib_figs": []
                    }
                    
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    response_text = response.get('output', 'Não foi possível processar a resposta.')

                    st.markdown(response_text)
                    
                    # --- Seção de Exibição ---
                    visuals = st.session_state.current_visualizations
                    
                    for fig in visuals.get("figures", []):
                        st.plotly_chart(fig, use_container_width=True)
                    for df_viz in visuals.get("dataframes", []):
                        st.dataframe(df_viz, use_container_width=True)
                    for fig_mpl in visuals.get("matplotlib_figs", []):
                        st.pyplot(fig_mpl)
                    
                    # --- Seção de Salvamento no Histórico ---
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "figures": visuals.get("figures", []),
                        "dataframes": visuals.get("dataframes", []),
                        "matplotlib_figs": visuals.get("matplotlib_figs", [])
                    })

                except Exception as e:
                    error_msg = f"❌ Ocorreu um erro: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "figures": [], "dataframes": [], "matplotlib_figs": []})