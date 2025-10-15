# 🤖 Agente de Análise Exploratória de Dados (EDA) com IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-red.svg) ![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-purple.svg) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)

Este projeto é um agente de Análise Exploratória de Dados (EDA) construído com Streamlit и LangChain. Ele utiliza o poder dos Grandes Modelos de Linguagem (LLMs) da OpenAI para interpretar perguntas em linguagem natural, executar análises complexas em dataframes Pandas e apresentar os resultados, incluindo visualizações de dados dinâmicas.

O objetivo é permitir que qualquer pessoa, independentemente do seu conhecimento em programação, possa extrair insights valiosos de seus dados através de uma interface de chat intuitiva.

###  Demo Rápida

![Demo do Agente EDA em Ação](https://github.com/backhenry/Agente-Analise-de-Dados-Exploratoria/blob/main/assets/Gravando 2025-10-15 010220.gif)

---

## ✨ Funcionalidades

O agente é equipado com um conjunto robusto de ferramentas para cobrir todo o ciclo de uma análise exploratória:

#### 📊 **Visão Geral e Descrição**
* **Resumo Completo (`get_data_summary`):** Fornece um perfil completo do dataset, incluindo tipos de dados, estatísticas descritivas e contagem de valores nulos.

#### 🧹 **Limpeza e Pré-processamento**
* **Tratamento de Nulos (`handle_missing_values`):** Preenche ou remove valores ausentes usando estratégias como média, mediana, moda ou um valor constante.
* **Remoção de Duplicatas (`handle_duplicates`):** Verifica e elimina linhas duplicadas.
* **Conversão de Tipos (`change_column_type`):** Altera o tipo de dado de colunas (ex: de `object` para `datetime`).

#### 🛠️ **Engenharia de Features**
* **Criação de Colunas com Fórmulas (`create_feature_from_math`):** Cria novas colunas a partir de operações matemáticas entre colunas existentes (ex: `preco * quantidade`).
* **Agrupamento de Dados Numéricos (`bin_numerical_column`):** Transforma uma variável numérica contínua em categórica (ex: agrupar idades em faixas etárias).

#### 📈 **Visualização de Dados**
* **Histogramas, Gráficos de Barras, Dispersão e Mapas de Calor.**
* **Nuvem de Palavras (`plot_word_cloud`):** Ideal para análise de texto.
* **Matriz de Gráficos (`plot_pair_plot`):** Permite visualizar a relação entre múltiplas variáveis de uma só vez.

#### 🧠 **Análise Avançada e Estatística**
* **Agregações Complexas (`get_aggregated_data`):** Realiza agrupamentos (`GROUP BY`) para calcular médias, somas, contagens, etc., e pode plotar o resultado.
* **Tabelas Dinâmicas (`create_pivot_table`):** Cria tabelas cruzadas para análises multivariadas.
* **Detecção de Outliers (`detect_outliers`):** Utiliza o algoritmo Isolation Forest для encontrar anomalias.
* **Teste T de Student (`perform_t_test`):** Compara as médias de dois grupos para verificar se a diferença é estatisticamente significante.

#### 📤 **Utilidades**
* **Exportação de Dados (`export_dataframe`):** Salva o dataframe (após as limpezas e modificações) em um arquivo `.csv`.

---

## 🚀 Tecnologias Utilizadas

* **Frontend:** Streamlit
* **Orquestração do Agente:** LangChain
* **Modelo de Linguagem:** OpenAI GPT-4o-mini (ou outro modelo compatível)
* **Manipulação de Dados:** Pandas
* **Visualização:** Plotly, Seaborn, Matplotlib
* **Análise Estatística e ML:** Scikit-learn, SciPy

---

## ⚙️ Configuração e Instalação

Siga os passos abaixo para executar o projeto localmente.

**1. Clone o Repositório**
```bash
git clone [https://github.com/backhenry/Agente-Analise-de-Dados-Exploratoria.git](https://github.com/backhenry/Agente-Analise-de-Dados-Exploratoria.git)
cd Agente-Analise-de-Dados-Exploratoria
2. Crie um Ambiente Virtual (Recomendado)

Bash

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
3. Instale as Dependências

Bash

pip install -r requirements.txt
4. Configure sua Chave de API
Crie um arquivo chamado .env na raiz do projeto e adicione sua chave da OpenAI:

OPENAI_API_KEY='sk-sua-chave-secreta-aqui'
5. Execute a Aplicação

Bash

streamlit run app.py
A aplicação será aberta no seu navegador!

💬 Como Usar
Com a aplicação aberta, utilize a barra lateral para fazer o upload de um arquivo de dados (.csv ou .zip contendo um .csv).

Clique no botão "Carregar e Inicializar Agente".

Aguarde a mensagem de sucesso "✅ Agente pronto!".

Comece a fazer perguntas sobre seus dados na caixa de chat no final da página.

Exemplos de Prompts
Faça um resumo completo dos dados.

Preencha os valores ausentes na coluna 'idade' com a mediana.

Crie uma coluna 'despesa_total' que seja a multiplicação de 'quantidade' por 'preco_unitario'.

Qual a taxa de sobrevivência por sexo? Gere um gráfico de barras.

A tarifa paga por quem sobreviveu é estatisticamente diferente da tarifa de quem não sobreviveu? Use as colunas 'survived' e 'fare' com grupos 1 e 0.

Crie uma tabela dinâmica mostrando a idade média, usando 'sex' como índice e 'pclass' como colunas.

🗺️ Melhorias Futuras
[ ] Suporte a mais formatos de arquivo (Excel, Parquet).

[ ] Adicionar mais testes estatísticos (ANOVA, Qui-quadrado).

[ ] Implementar um sistema de cache para otimizar respostas repetidas.

[ ] Adicionar a capacidade de salvar e carregar o histórico de conversas.

[ ] Fazer o deploy da aplicação na Streamlit Community Cloud.

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
