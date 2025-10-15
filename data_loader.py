# /data_loader.py

import io
import zipfile
import re
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner="Analisando e limpando o arquivo...")
def load_and_process_data(uploaded_file):
    """Carrega dados de CSV ou ZIP, tentando diferentes formatos e limpando os nomes das colunas."""
    if uploaded_file is None:
        return {"status": "error", "message": "Nenhum arquivo recebido."}

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    bio = io.BytesIO(file_bytes)

    def clean_column_names(df):
        new_columns = []
        for c in df.columns:
            c = str(c).lower().strip()
            c = re.sub(r'[\s.\-\/]+', '_', c)  # Substitui espaços e outros separadores por _
            c = re.sub(r'[^a-z0-9_]', '', c)    # Remove caracteres não alfanuméricos
            new_columns.append(c)
        df.columns = new_columns
        return df

    def read_csv_flexible(file_buffer):
        try:
            file_buffer.seek(0)
            return pd.read_csv(file_buffer, sep=',', decimal='.', thousands=None)
        except Exception:
            pass

        try:
            file_buffer.seek(0)
            return pd.read_csv(file_buffer, sep=';', decimal=',', thousands='.')
        except Exception as e:
            raise ValueError(f"Não foi possível analisar o CSV. Verifique o delimitador (',' ou ';') e o formato decimal ('.' ou ','). Erro: {e}")

    try:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(bio, "r") as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_names:
                    return {"status": "error", "message": "O arquivo ZIP não contém nenhum arquivo CSV."}
                with z.open(csv_names[0]) as f:
                    csv_content = io.BytesIO(f.read())
                    df = read_csv_flexible(csv_content)
        elif filename.lower().endswith(".csv"):
            df = read_csv_flexible(bio)
        else:
            return {"status": "error", "message": "Formato de arquivo não suportado. Por favor, envie um arquivo CSV ou ZIP."}

        df = clean_column_names(df)
        return {"status": "success", "df": df, "message": f"Arquivo '{filename}' carregado com sucesso."}

    except Exception as e:
        return {"status": "error", "message": f"Erro fatal ao processar o arquivo: {e}"}