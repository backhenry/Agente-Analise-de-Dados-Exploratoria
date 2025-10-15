# /tools.py

import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# ==============================================================================
# NOTA DE ARQUITETURA:
# Todas as ferramentas interagem com o Streamlit Session State de duas formas:
# 1. Lendo o DataFrame principal: `st.session_state.df`
# 2. Escrevendo visualizações em `st.session_state.current_visualizations`
#    para que a UI possa exibi-las após a execução da ferramenta.
# ==============================================================================


# ==============================================================================
# SEÇÃO 1: VISÃO GERAL E DESCRIÇÃO DOS DADOS
# ==============================================================================

def get_data_summary(*args, **kwargs):
    """
    Fornece um resumo completo do DataFrame, incluindo informações gerais, estatísticas descritivas,
    contagem de valores nulos e valores únicos por coluna. É a melhor ferramenta para começar
    qualquer análise. Não requer argumentos.
    """
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    try:
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        
        desc_stats = df.describe(include=np.number).round(2)
        st.session_state.current_visualizations["dataframes"].append(desc_stats)
        
        null_counts = df.isnull().sum()
        null_percentage = (null_counts / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Valores Nulos': null_counts,
            '% Nulos': null_percentage
        }).sort_values(by='Valores Nulos', ascending=False)
        st.session_state.current_visualizations["dataframes"].append(missing_df[missing_df['Valores Nulos'] > 0])
        
        summary_message = (
            f"### Resumo Completo do DataFrame\n\n"
            f"**1. Formato:**\n- O DataFrame possui {df.shape[0]} linhas e {df.shape[1]} colunas.\n\n"
            f"**2. Tipos de Dados e Entradas Não Nulas:**\n```\n{info_str}\n```\n"
            f"**3. Estatísticas Descritivas (Colunas Numéricas):**\n- A tabela de estatísticas foi gerada para visualização.\n\n"
            f"**4. Análise de Valores Ausentes:**\n- A tabela de valores ausentes foi gerada para visualização.\n"
        )
        return {"status": "success", "message": summary_message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar resumo dos dados: {e}"}

# ==============================================================================
# SEÇÃO 2: LIMPEZA E PRÉ-PROCESSAMENTO
# ==============================================================================

def handle_missing_values(instruction_str: str, *args, **kwargs):
    """
    Preenche ou remove valores ausentes (NaN) de uma coluna.
    Formato: 'coluna, estrategia'. 
    Estratégias válidas: 'remover', 'media', 'mediana', 'moda', ou um valor específico (ex: 0, 'Desconhecido').
    """
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    parts = [p.strip().lower() for p in instruction_str.split(',')]
    if len(parts) != 2:
        return {"status": "error", "message": "Formato inválido. Use: 'coluna, estrategia'."}
        
    col, strategy = parts[0], parts[1]
    
    if col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}
        
    initial_nulls = df[col].isnull().sum()
    if initial_nulls == 0:
        return {"status": "success", "message": f"A coluna '{col}' não possui valores ausentes."}

    try:
        df_copy = df.copy()
        if strategy == 'remover':
            df_copy.dropna(subset=[col], inplace=True)
            message = f"Foram removidas {initial_nulls} linhas com valores ausentes na coluna '{col}'."
        elif strategy in ['media', 'mediana']:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                return {"status": "error", "message": f"Estratégia '{strategy}' só pode ser usada em colunas numéricas."}
            fill_value = df_copy[col].mean() if strategy == 'media' else df_copy[col].median()
            df_copy[col].fillna(fill_value, inplace=True)
            message = f"Foram preenchidos {initial_nulls} valores ausentes em '{col}' com a {strategy} ({fill_value:.2f})."
        elif strategy == 'moda':
            fill_value = df_copy[col].mode()[0]
            df_copy[col].fillna(fill_value, inplace=True)
            message = f"Foram preenchidos {initial_nulls} valores ausentes em '{col}' com a moda ('{fill_value}')."
        else:
            try:
                fill_value = pd.Series([strategy]).astype(df_copy[col].dtype).iloc[0]
                df_copy[col].fillna(fill_value, inplace=True)
                message = f"Foram preenchidos {initial_nulls} valores ausentes em '{col}' com o valor '{fill_value}'."
            except (ValueError, TypeError):
                 return {"status": "error", "message": f"Não foi possível usar '{strategy}' como valor de preenchimento para a coluna '{col}'."}

        st.session_state.df = df_copy
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao tratar valores ausentes: {e}"}

def handle_duplicates(*args, **kwargs):
    """
    Verifica a existência de linhas duplicadas no DataFrame e as remove.
    Não requer argumentos.
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    try:
        duplicate_count = df.duplicated().sum()
        if duplicate_count == 0:
            return {"status": "success", "message": "Não foram encontradas linhas duplicadas no DataFrame."}
        
        df.drop_duplicates(inplace=True)
        st.session_state.df = df
        
        return {
            "status": "success",
            "message": f"Foram encontradas e removidas {duplicate_count} linhas duplicadas. O DataFrame agora possui {len(df)} linhas."
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao lidar com duplicatas: {e}"}

def change_column_type(instruction_str: str, *args, **kwargs):
    """
    Altera o tipo de dados (dtype) de uma coluna.
    Formato: 'coluna, novo_tipo'.
    Tipos comuns: 'int', 'float', 'str' (ou 'object'), 'datetime'.
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    parts = [p.strip().lower() for p in instruction_str.split(',')]
    if len(parts) != 2:
        return {"status": "error", "message": "Formato inválido. Use: 'coluna, novo_tipo'."}
    
    col, new_type = parts[0], parts[1]

    if col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}

    try:
        df_copy = df.copy()
        original_type = df_copy[col].dtype
        
        if new_type in ['int', 'integer']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')
        elif new_type in ['float', 'numeric']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        elif new_type in ['str', 'string', 'object']:
            df_copy[col] = df_copy[col].astype(str)
        elif new_type in ['datetime', 'date']:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        else:
            return {"status": "error", "message": f"Tipo '{new_type}' não suportado. Use 'int', 'float', 'str' ou 'datetime'."}
        
        st.session_state.df = df_copy
        return {
            "status": "success",
            "message": f"O tipo da coluna '{col}' foi alterado de '{original_type}' para '{df_copy[col].dtype}'."
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao converter tipo da coluna '{col}': {e}"}

# ==============================================================================
# SEÇÃO 3: ENGENHARIA DE FEATURES
# ==============================================================================

def create_feature_from_math(instruction_str: str, *args, **kwargs):
    """
    Cria uma nova coluna executando uma operação matemática. Usa a função `df.eval()`.
    Formato: 'nome_nova_coluna = expressao_matematica'
    Exemplo: 'total_value = quantity * price'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    if '=' not in instruction_str:
        return {"status": "error", "message": "Formato inválido. Use: 'nova_coluna = expressao'."}

    new_col, expression = instruction_str.split('=', 1)
    new_col = new_col.strip()
    
    try:
        df_copy = df.copy()
        df_copy[new_col] = df_copy.eval(expression)
        st.session_state.df = df_copy
        return {
            "status": "success",
            "message": f"Coluna '{new_col}' criada com sucesso a partir da expressão: '{expression}'."
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao avaliar a expressão: {e}. Verifique se os nomes das colunas estão corretos."}

def bin_numerical_column(instruction_str: str, *args, **kwargs):
    """
    Discretiza uma coluna numérica em N grupos (bins) de igual frequência (quantis).
    Formato: 'coluna_numerica, nome_nova_coluna, numero_de_grupos'
    Exemplo: 'age, age_group, 4'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    parts = [p.strip() for p in instruction_str.split(',')]
    if len(parts) != 3:
        return {"status": "error", "message": "Formato inválido. Use: 'coluna, nova_coluna, n_grupos'."}
    
    col, new_col, n_bins_str = parts
    
    try:
        n_bins = int(n_bins_str)
        df_copy = df.copy()
        df_copy[new_col] = pd.qcut(df_copy[col], q=n_bins, labels=False, duplicates='drop')
        st.session_state.df = df_copy
        return {
            "status": "success",
            "message": f"Coluna '{col}' foi agrupada em {n_bins} bins na nova coluna '{new_col}'."
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao criar bins: {e}. Verifique se a coluna é numérica."}

# ==============================================================================
# SEÇÃO 4: ANÁLISE E VISUALIZAÇÃO
# ==============================================================================

def plot_histogram(column: str, *args, **kwargs):
    """
    Gera um histograma para visualizar a distribuição de uma única coluna numérica.
    Formato: 'nome_da_coluna'
    """
    df = st.session_state.df
    col = column.lower().strip()
    if df is None or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {"status": "error", "message": f"Coluna '{col}' não encontrada ou não é numérica."}
    try:
        fig = px.histogram(df, x=col, title=f"Distribuição de {col}", marginal="box")
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": f"Histograma gerado para '{col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar histograma: {e}"}

def plot_bar_chart(column: str, *args, **kwargs):
    """
    Gera um gráfico de barras para visualizar a frequência de cada categoria em uma única coluna categórica.
    Formato: 'nome_da_coluna'
    """
    df = st.session_state.df
    col = column.lower().strip()
    if df is None or col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}
    
    if pd.api.types.is_numeric_dtype(df[col]):
        return {"status": "error", "message": f"Coluna '{col}' é numérica. Use plot_histogram para ver sua distribuição."}

    try:
        counts = df[col].value_counts().reset_index().head(20) # Limita a 20 categorias para legibilidade
        counts.columns = [col, 'count']
        fig = px.bar(counts, x=col, y='count', title=f'Contagem de Frequência para {col}', text_auto=True)
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": f"Gráfico de barras de frequência gerado para '{col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar gráfico de barras: {e}"}

def plot_scatter(instruction_str: str, *args, **kwargs):
    """
    Gera um gráfico de dispersão (scatter plot) para visualizar a relação entre duas colunas numéricas.
    Formato: 'coluna_x, coluna_y'
    """
    df = st.session_state.df
    col_names = [c.strip().lower() for c in instruction_str.split(',') if c]
    
    if df is None or len(col_names) != 2:
        return {"status": "error", "message": "Forneça exatamente duas colunas numéricas (x, y)."}
    
    x_col, y_col = col_names[0], col_names[1]
    if not all(c in df.columns for c in [x_col, y_col]):
        return {"status": "error", "message": f"Colunas '{x_col}' ou '{y_col}' não encontradas."}
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        return {"status": "error", "message": "Ambas as colunas para o gráfico de dispersão devem ser numéricas."}

    try:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"Relação entre {x_col} e {y_col}", trendline="ols")
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": f"Gráfico de dispersão gerado para '{x_col}' vs '{y_col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar gráfico de dispersão: {e}"}

def plot_correlation_heatmap(*args, **kwargs):
    """
    Gera um mapa de calor (heatmap) da correlação de Pearson entre todas as colunas numéricas do DataFrame.
    Não requer argumentos.
    """
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "São necessárias pelo menos duas colunas numéricas para gerar uma matriz de correlação."}
    
    try:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Matriz de Correlação', color_continuous_scale='RdBu_r')
        st.session_state.current_visualizations["figures"].append(fig)
        return {"status": "success", "message": "Mapa de calor da correlação gerado."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar mapa de calor de correlação: {e}"}

def plot_word_cloud(column: str, *args, **kwargs):
    """
    Gera uma nuvem de palavras (word cloud) a partir de uma coluna de texto.
    Formato: 'nome_da_coluna'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    col = column.strip().lower()
    if col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}

    try:
        text = ' '.join(df[col].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        st.session_state.current_visualizations.setdefault("matplotlib_figs", []).append(fig)
        
        return {"status": "success", "message": f"Nuvem de palavras gerada para a coluna '{col}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar nuvem de palavras: {e}"}

def plot_pair_plot(instruction_str: str, *args, **kwargs):
    """
    Gera uma matriz de gráficos de dispersão (pair plot) para visualizar a relação entre múltiplas colunas.
    Formato: 'col1, col2, col3; coluna_cor (opcional)'
    Exemplo: 'sepal_length, sepal_width; species'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    parts = [p.strip() for p in instruction_str.split(';')]
    cols_to_plot = [c.strip() for c in parts[0].split(',')]
    hue_col = parts[1] if len(parts) > 1 else None

    try:
        fig = sns.pairplot(df, vars=cols_to_plot, hue=hue_col)
        st.session_state.current_visualizations.setdefault("matplotlib_figs", []).append(fig)
        return {"status": "success", "message": f"Pair plot gerado para as colunas: {', '.join(cols_to_plot)}."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar pair plot: {e}"}

# ==============================================================================
# SEÇÃO 5: ANÁLISE AVANÇADA, GRUPOS E ESTATÍSTICA
# ==============================================================================

def get_aggregated_data(instruction_str: str, *args, **kwargs):
    """
    Ferramenta poderosa para agrupar dados e calcular métricas. Pode também gerar um gráfico do resultado.
    Formato: 'coluna_de_grupo; coluna_numerica, agregacao; tipo_de_grafico (opcional)'
    Agregações: 'mean', 'sum', 'count', 'median', 'min', 'max'.
    Gráficos: 'bar'
    Exemplo 1 (apenas dados): 'sex; survived, mean'
    Exemplo 2 (dados e gráfico): 'pclass; fare, median; bar'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    try:
        parts = [p.strip().lower() for p in instruction_str.split(';')]
        plot_type = None
        if len(parts) == 3:
            plot_type = parts[2]
        
        group_col = parts[0]
        agg_parts = [p.strip().lower() for p in parts[1].split(',')]
        
        num_col, agg_func = agg_parts[0], agg_parts[1]

        if group_col not in df.columns or num_col not in df.columns:
            return {"status": "error", "message": f"Coluna de grupo '{group_col}' ou numérica '{num_col}' não encontrada."}
            
        agg_df = df.groupby(group_col)[num_col].agg(agg_func).round(2).reset_index()
        
        message = f"Dados agregados com sucesso. A tabela de resultados é:\n\n{agg_df.to_markdown()}"
        
        if plot_type == 'bar':
            fig = px.bar(agg_df, x=group_col, y=num_col, title=f'{agg_func.capitalize()} de {num_col} por {group_col}', text_auto=True)
            st.session_state.current_visualizations["figures"].append(fig)
            message += f"\n\nUm gráfico de barras mostrando o resultado também foi gerado."
            
        st.session_state.current_visualizations["dataframes"].append(agg_df)
        return {"status": "success", "message": message}

    except Exception as e:
        return {"status": "error", "message": f"Erro na agregação: {e}. Verifique o formato: 'grupo; numerica, agg; bar (opcional)'."}

def create_pivot_table(instruction_str: str, *args, **kwargs):
    """
    Cria uma tabela dinâmica (pivot table) para cruzar três variáveis.
    Formato: 'coluna_indice, coluna_agrupamento, coluna_valor, funcao_agregacao'
    Exemplo: 'sex, pclass, fare, mean' -> mostra a tarifa média (fare) para cada sexo e classe (pclass).
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    parts = [p.strip().lower() for p in instruction_str.split(',')]
    if len(parts) != 4:
        return {"status": "error", "message": "Formato inválido. Use: 'indice, colunas, valores, agregacao'."}

    index_col, columns_col, values_col, agg_func = parts

    if not all(c in df.columns for c in [index_col, columns_col, values_col]):
        return {"status": "error", "message": "Uma ou mais colunas especificadas não foram encontradas."}

    try:
        pivot_df = pd.pivot_table(df, values=values_col, index=index_col, columns=columns_col, aggfunc=agg_func).round(2)
        st.session_state.current_visualizations["dataframes"].append(pivot_df)
        
        return {
            "status": "success",
            "message": f"Tabela dinâmica criada com sucesso. Os resultados estão exibidos na tabela abaixo.\n\n{pivot_df.to_markdown()}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao criar tabela dinâmica: {e}"}
def perform_t_test(instruction_str: str, *args, **kwargs):
    """
    Realiza um Teste T de Student para comparar as médias de dois grupos independentes.
    Formato: 'coluna_categorica, nome_grupo1, nome_grupo2, coluna_numerica'
    Exemplo: 'smoker, yes, no, charges'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}

    parts = [p.strip() for p in instruction_str.split(',')]
    if len(parts) != 4:
        return {"status": "error", "message": "Formato inválido. Use: 'col_cat, grupo1, grupo2, col_num'."}
    
    cat_col, group1_str, group2_str, num_col = parts

    try:
        # --- INÍCIO DA CORREÇÃO ---
        # Pega o tipo de dado da coluna alvo (ex: int64 para 'survived')
        target_dtype = df[cat_col].dtype
        
        # Converte as strings '1' e '0' para o tipo de dado da coluna alvo (int64)
        group1 = pd.Series([group1_str]).astype(target_dtype).iloc[0]
        group2 = pd.Series([group2_str]).astype(target_dtype).iloc[0]
        # --- FIM DA CORREÇÃO ---

        sample1 = df[df[cat_col] == group1][num_col].dropna()
        sample2 = df[df[cat_col] == group2][num_col].dropna()

        if len(sample1) < 2 or len(sample2) < 2:
            return {"status": "error", "message": f"Um ou ambos os grupos não possuem pelo menos duas amostras. Grupo '{group1}': {len(sample1)} amostras, Grupo '{group2}': {len(sample2)} amostras."}

        stat, p_value = ttest_ind(sample1, sample2)
        
        alpha = 0.05
        conclusion = "A diferença entre as médias é estatisticamente significante." if p_value < alpha else "A diferença entre as médias NÃO é estatisticamente significante."

        message = (f"**Resultado do Teste T para a variável '{num_col}':**\n"
                   f"- Média do Grupo '{group1}': {sample1.mean():.2f}\n"
                   f"- Média do Grupo '{group2}': {sample2.mean():.2f}\n"
                   f"- Estatística T: {stat:.4f}\n"
                   f"- P-valor: {p_value:.4f}\n\n"
                   f"**Conclusão (com alpha=0.05):** {conclusion}")
        
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao executar o Teste T: {e}."}

def detect_outliers(columns_str: str, *args, **kwargs):
    """
    Detecta outliers em uma ou mais colunas numéricas usando o algoritmo IsolationForest.
    Formato: 'coluna1, coluna2, ...'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame carregado."}
        
    feature_cols = [c.strip().lower() for c in columns_str.split(',') if c]
    if not feature_cols:
        return {"status": "error", "message": "Forneça pelo menos uma coluna numérica."}

    try:
        X = df[feature_cols].select_dtypes(include=np.number).dropna()
        if X.empty:
            return {"status": "error", "message": "As colunas fornecidas não são numéricas ou estão vazias."}

        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(X)
        
        out_idx = X.index[labels == -1]
        sample = df.loc[out_idx]
        
        msg = (f"Foram detectados **{len(out_idx)}** outliers "
               f"({(len(out_idx) / len(X) * 100):.2f}% dos dados analisados) "
               f"usando IsolationForest nas colunas `{', '.join(feature_cols)}`.")
        
        if not sample.empty:
            st.session_state.current_visualizations["dataframes"].append(sample.head(10))
            msg += "\nUma amostra de 10 outliers é exibida na tabela."

        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": f"Erro na detecção de outliers: {e}"}

# ==============================================================================
# SEÇÃO 6: UTILIDADES
# ==============================================================================

def export_dataframe(filename: str, *args, **kwargs):
    """
    Salva o DataFrame atual em um arquivo CSV. O arquivo será salvo localmente onde o script está rodando.
    Formato: 'nome_do_arquivo.csv'
    """
    df = st.session_state.df
    if df is None: return {"status": "error", "message": "Nenhum DataFrame para exportar."}
    
    fname = filename.strip()
    if not fname.endswith('.csv'):
        return {"status": "error", "message": "O nome do arquivo deve terminar com .csv"}
    
    try:
        df.to_csv(fname, index=False)
        return {"status": "success", "message": f"DataFrame salvo com sucesso como '{fname}'."}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao salvar arquivo: {e}"}