# Agente Inteligente de Análise de Ofertas Primárias

Projeto desenvolvido para o case do BTG Pactual - Inteli Academy.


## Objetivo

O projeto tem como objetivo automatizar a coleta, análise e contextualização de ofertas primárias do mercado financeiro, permitindo identificar padrões, discrepâncias e oportunidades entre diferentes instituições.

O sistema utiliza agentes inteligentes para integrar:

- coleta automatizada de dados;
- processamento estruturado;
- análise contextual;
- insights baseados em informações macroeconômicas.

---

## Tecnologias Utilizadas

- Python
- LangChain
- Pandas
- Streamlit
- Web Scraping
- APIs Financeiras

---

## Estrutura do Projeto

```bash
CASEBTG/
│
├── data/
│   ├── cvm/
│   ├── cvm_sre/
│   └── scraped/
│
├── src/
│   ├── agent/
│   │   └── agent.py
│   │
│   ├── data_ingestion/
│   │   ├── cvm_csv.py
│   │   ├── cvm_sre.py
│   │   └── scraping.py
│   │
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Funcionalidades

- Coleta automatizada de dados da CVM
- Processamento de dados estruturados
- Web scraping de ofertas
- Agente inteligente para análise
- Consolidação de informações financeiras
- Contextualização de mercado

---

## Pipeline do Projeto

```text
Coleta de Dados
       ↓
Tratamento e Padronização
       ↓
Análise Contextual
       ↓
Agente Inteligente
       ↓
Insights e Relatórios
```

---

## Como Executar

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
```

---

### 2. Crie o ambiente virtual

```bash
python -m venv venv
```

---

### 3. Ative o ambiente

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 5. Configure variáveis de ambiente

Crie um arquivo `.env`

```env
OPENAI_API_KEY=sua_chave
```

---

### 6. Execute a aplicação

```bash
python src/app.py
```

ou

```bash
streamlit run src/app.py
```

---

## Arquitetura do Agente

O agente foi desenvolvido utilizando LangChain para orquestração das etapas de:

- coleta;
- análise;
- interpretação contextual;
- geração de insights.

---

## Melhorias Futuras

- Integração com APIs em tempo real
- Dashboard interativo avançado
- Sistema multi-agente
- RAG com notícias e relatórios

---

### Raphaela Rodrigues Luvizotto