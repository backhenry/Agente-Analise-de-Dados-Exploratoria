# 🤖 Agente EDA - Análise Exploratória de Dados com IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agente-fraude.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.6-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agente autônomo inteligente para análise exploratória de dados (EDA) especializado em detecção de fraudes financeiras. Utiliza LangChain, Streamlit e OpenAI para fornecer insights automatizados através de uma interface de chat natural.

## 🎯 Funcionalidades

### 📊 Análises Automatizadas
- **Estatísticas Descritivas**: Tipos de dados, contagens, médias, desvio padrão e quartis
- **Detecção de Outliers**: Algoritmo Isolation Forest para identificar anomalias
- **Análise de Correlações**: Matriz de correlação visual com heatmap interativo
- **Clustering**: Segmentação de dados com K-Means

### 📈 Visualizações Interativas
- Histogramas de distribuição
- Gráficos de dispersão (scatter plots)
- Mapas de calor de correlação
- Gráficos coloridos por classe

### 🧠 Inteligência Artificial
- Interface de chat natural powered by GPT-4o-mini
- Memória conversacional para contexto entre perguntas
- Execução autônoma de ferramentas especializadas
- Geração de insights e conclusões sintetizadas

## 🚀 Quick Start

### Pré-requisitos

- Python 3.10 ou superior
- Chave de API da OpenAI ([obtenha aqui](https://platform.openai.com/api-keys))
- Git

### Instalação

1. **Clone o repositório**
```

git clone https://github.com/backhenry/Agente-Fraude.git
cd Agente-Fraude

```

2. **Crie um ambiente virtual**
```

python -m venv venv
source venv/bin/activate  \# No Windows: venv\Scripts\activate

```

3. **Instale as dependências**
```

pip install -r requirements.txt

```

4. **Configure a chave da API**

Crie um arquivo `.env` na raiz do projeto:
```

cp .env.example .env

```

Edite o arquivo `.env` e adicione sua chave:
```

OPENAI_API_KEY=sk-proj-sua-chave-aqui

```

5. **Execute a aplicação**
```

streamlit run app.py

```

A aplicação abrirá automaticamente em `http://localhost:8501`

## 📖 Como Usar

### 1. Upload de Dados
Na barra lateral, faça upload de um arquivo CSV ou ZIP contendo um CSV:
- **Formatos suportados**: `.csv`, `.zip`
- **Tamanho recomendado**: Até 200MB
- **Estrutura**: Qualquer dataset com colunas numéricas

### 2. Inicialize o Agente
Clique no botão **"🚀 Carregar e Inicializar Agente"**

### 3. Faça Perguntas
Digite perguntas em linguagem natural no chat:

#### Exemplos de Perguntas
```

📊 Análise Inicial

- "Quais são os tipos de dados e as estatísticas descritivas?"
- "Quantas linhas e colunas tem o dataset?"
- "Existem valores nulos?"

🔍 Detecção de Anomalias

- "Detecte outliers nos dados"
- "Quais transações são suspeitas?"

📈 Visualizações

- "Gere o mapa de calor das correlações"
- "Crie um histograma da coluna amount"
- "Faça um gráfico de dispersão entre time e amount"

🧮 Análise Avançada

- "Agrupe os dados em 3 clusters"
- "Quais variáveis estão mais correlacionadas?"

💡 Conclusões

- "Qual a conclusão geral das análises que fizemos?"
- "Resuma os principais insights"

```

## 🏗️ Arquitetura

```

┌─────────────────────────────────────────────┐
│          Interface (Streamlit)              │
│  Chat UI │ Upload │ Visualizações           │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│         Agente (LangChain)                  │
│  AgentExecutor │ Memory │ System Prompt     │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│              Ferramentas                    │
│  ├─ show_descriptive_stats()               │
│  ├─ generate_histogram()                   │
│  ├─ generate_correlation_heatmap()         │
│  ├─ generate_scatter_plot()                │
│  ├─ detect_outliers_isolation_forest()     │
│  └─ find_clusters_kmeans()                 │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│          Camada de Dados                    │
│        Pandas │ NumPy │ Scikit-learn        │
└─────────────────────────────────────────────┘

```

## 🛠️ Tecnologias

| Categoria | Tecnologia | Versão | Propósito |
|-----------|-----------|--------|-----------|
| **Interface** | Streamlit | 1.31.0 | UI web interativa |
| **IA** | LangChain | 0.1.6 | Orquestração de agente |
| **LLM** | OpenAI GPT-4o-mini | - | Modelo de linguagem |
| **Dados** | Pandas | 2.1.4 | Manipulação de dados |
| **Visualização** | Plotly | 5.18.0 | Gráficos interativos |
| **ML** | Scikit-learn | 1.4.0 | Algoritmos de ML |
| **Config** | Python-dotenv | 1.0.1 | Gerenciamento de ambiente |

## 📁 Estrutura do Projeto

```

Agente-Fraude/
│
├── app.py                  \# Aplicação principal Streamlit
├── requirements.txt        \# Dependências Python
├── .env.example           \# Template de configuração
├── .gitignore             \# Arquivos ignorados pelo Git
├── README.md              \# Este arquivo
│
└── docs/                  \# Documentação adicional (opcional)
├── screenshots/       \# Capturas de tela
└── examples/          \# Datasets de exemplo

```

## 🔒 Segurança

### Boas Práticas Implementadas

✅ **Variáveis de Ambiente**: Chaves API nunca no código  
✅ **.gitignore**: `.env` excluído do controle de versão  
✅ **Template**: `.env.example` fornecido sem valores reais  
✅ **Validação**: Verificação de chave antes da execução  
✅ **Secrets**: Deploy usa secrets da plataforma  

### ⚠️ Importante

- **Nunca** commite o arquivo `.env`
- **Nunca** compartilhe sua chave API publicamente
- **Rotacione** chaves periodicamente
- **Monitore** uso da API OpenAI

## 🌐 Deploy

### Streamlit Community Cloud (Gratuito)

1. Crie uma conta em [share.streamlit.io](https://share.streamlit.io)
2. Conecte seu repositório GitHub
3. Configure os Secrets:
```

OPENAI_API_KEY = "sua-chave-aqui"

```
4. Clique em "Deploy"

### Outras Opções

- **Hugging Face Spaces**: Deploy gratuito com GPU opcional
- **Railway**: $5 crédito gratuito mensal
- **Heroku**: Planos pagos, fácil configuração
- **AWS/Azure/GCP**: Para produção em larga escala

## 📊 Exemplo de Dataset

O agente funciona melhor com datasets que contenham:

- Colunas numéricas para análise estatística
- Dados temporais para análise de séries
- Classificação binária/multi-classe (opcional)
- Formato CSV limpo e estruturado

### Dataset de Exemplo: Credit Card Fraud Detection

```

time,v1,v2,v3,...,v28,amount,class
0,-1.359807134,-0.072781173,2.536346738,...,-0.189114844,149.62,0
1,-1.358354062,0.877736754,1.548717847,...,-0.066794157,2.69,0
...

```

[Download dataset de exemplo](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Diretrizes

- Siga o style guide PEP 8
- Adicione testes para novas funcionalidades
- Atualize a documentação conforme necessário
- Mantenha commits pequenos e descritivos

## 🐛 Reportando Bugs

Encontrou um bug? Abra uma [issue](https://github.com/backhenry/Agente-Fraude/issues) com:

- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs atual
- Screenshots (se aplicável)
- Informações do ambiente (OS, Python version)

## 📝 Roadmap

### Em Desenvolvimento
- [ ] Exportação de relatórios em PDF
- [ ] Suporte para múltiplos datasets
- [ ] Integração com bancos de dados SQL

### Futuro
- [ ] Mais algoritmos de ML (regressão, classificação)
- [ ] Análise de séries temporais
- [ ] Dashboard customizável
- [ ] Autenticação de usuários
- [ ] API REST

## 📚 Documentação Adicional

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Plotly Documentation](https://plotly.com/python/)

## 👥 Autor

**Henrique Martins** - [GitHub](https://github.com/backhenry) - [LinkedIn](https://www.linkedin.com/in/rick-b-martins/)


## 🙏 Agradecimentos

- [LangChain](https://www.langchain.com/) pela framework de agentes
- [Streamlit](https://streamlit.io/) pela plataforma de deploy gratuita
- [OpenAI](https://openai.com/) pelos modelos de linguagem
- Comunidade open source pelos pacotes utilizados

## 📧 Contato

Dúvidas ou sugestões? Entre em contato:

- **Email**: henrique03martins@gmail.com
- **LinkedIn**: [rick-b-martins](https://www.linkedin.com/in/rick-b-martins/)
- **GitHub**: [@backhenry](https://github.com/backhenry)

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!

**Desenvolvido com ❤️ usando Python e IA**
```
