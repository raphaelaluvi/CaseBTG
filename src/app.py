import sys
import time
import json
import datetime
import traceback
from pathlib import Path
from io import StringIO

import pandas as pd
import streamlit as st

import sys
try:
    from agent import (
        build_agent,
        ranking_ofertas_fii,
        get_dados,
    )

    AGENT_IMPORT_OK = True
    AGENT_IMPORT_ERROR = None

except Exception as e:

    AGENT_IMPORT_OK = False
    AGENT_IMPORT_ERROR = str(e)

    print("Erro importando agent.py:")
    print(e)


SRC_ROOT = Path(__file__).resolve().parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.agent import (
    build_agent,
    ranking_ofertas_fii,
    _DADOS,
    get_dados,
)
# ─── Configuração da Página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="LigaAI — Mercado Primário",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "LigaAI — Agente de Análise de Ofertas Primárias | Powered by LangGraph + Groq",
    },
)

# ─── CSS / Tema ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Imports de fonte ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&display=swap');

/* ── Variáveis ── */
:root {
    --bg-primary:    #0a0e17;
    --bg-secondary:  #0f1520;
    --bg-card:       #131c2e;
    --bg-elevated:   #1a2540;
    --bg-input:      #111827;
    --border:        #1e2d47;
    --border-bright: #2a3f5f;
    --accent-blue:   #1d6fa4;
    --accent-cyan:   #00b4d8;
    --accent-green:  #00c896;
    --accent-yellow: #f0b429;
    --accent-red:    #e53e3e;
    --text-primary:  #e8edf5;
    --text-secondary:#8fa3bf;
    --text-muted:    #4a6080;
    --font-main:     'IBM Plex Sans', sans-serif;
    --font-mono:     'IBM Plex Mono', monospace;
}

/* ── Reset global ── */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-main) !important;
}

/* ── Header da página ── */
[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

/* ── Containers principais ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── Cards de métricas customizados ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.metric-card .label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 3px;
}

/* ── Status badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-family: var(--font-mono);
    margin: 2px;
}
.badge-ok    { background: rgba(0,200,150,.15); border: 1px solid rgba(0,200,150,.4); color: var(--accent-green); }
.badge-warn  { background: rgba(240,180,41,.12); border: 1px solid rgba(240,180,41,.4); color: var(--accent-yellow); }
.badge-err   { background: rgba(229,62,62,.12); border: 1px solid rgba(229,62,62,.4); color: var(--accent-red); }
.badge-info  { background: rgba(0,180,216,.12); border: 1px solid rgba(0,180,216,.4); color: var(--accent-cyan); }

/* ── Sugestões rápidas ── */
.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 20px;
}
.suggestion-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    cursor: pointer;
    transition: all .18s ease;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
}
.suggestion-card:hover {
    border-color: var(--accent-blue);
    background: var(--bg-elevated);
    color: var(--text-primary);
}
.suggestion-icon {
    font-size: 18px;
    margin-bottom: 6px;
    display: block;
}

/* ── Mensagens do chat ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0 24px;
    min-height: 200px;
}
.msg-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.msg-row.user   { flex-direction: row-reverse; }
.msg-avatar {
    width: 32px; height: 32px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    font-family: var(--font-mono);
    font-weight: 600;
}
.avatar-user  { background: var(--accent-blue); color: #fff; }
.avatar-agent { background: var(--bg-elevated); border: 1px solid var(--border-bright); color: var(--accent-cyan); }
.msg-bubble {
    max-width: 82%;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.65;
    position: relative;
}
.bubble-user {
    background: var(--accent-blue);
    color: #fff;
    border-bottom-right-radius: 2px;
}
.bubble-agent {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-bottom-left-radius: 2px;
}
.bubble-agent pre, .bubble-agent code {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    color: var(--accent-cyan);
}
.bubble-agent pre { padding: 10px 14px; overflow-x: auto; }
.msg-meta {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 5px;
    font-family: var(--font-mono);
    display: flex;
    gap: 10px;
    align-items: center;
}

/* ── Tool call badge ── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,180,216,.1);
    border: 1px solid rgba(0,180,216,.25);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    margin: 4px 0;
}

/* ── Input area ── */
.input-area {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, var(--bg-primary) 80%, transparent);
    padding: 16px 0 8px;
    z-index: 100;
}

/* ── Tabela de ranking ── */
.ranking-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ranking-table th {
    background: var(--bg-elevated);
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
}
.ranking-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
    vertical-align: middle;
}
.ranking-table tr:hover td { background: var(--bg-elevated); }
.score-bar {
    height: 4px; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
    margin-top: 3px;
}

/* ── Header customizado ── */
.page-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
    margin-bottom: 24px;
}
.page-title {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin: 0;
}
.page-title span { color: var(--accent-cyan); }
.page-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
    font-weight: 300;
}

/* ── Separador ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
}

/* ── Erro ── */
.error-box {
    background: rgba(229,62,62,.1);
    border: 1px solid rgba(229,62,62,.35);
    border-radius: 6px;
    padding: 14px 18px;
    color: #fc8181;
    font-size: 13px;
    margin: 12px 0;
}

/* ── Overrides Streamlit ── */
.stButton > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
    border-radius: 5px !important;
    font-family: var(--font-main) !important;
    font-size: 13px !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    background: rgba(0,180,216,.08) !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    font-size: 14px !important;
    caret-color: var(--accent-cyan);
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,180,216,.15) !important;
}
.stDataFrame, .stDataFrame * {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
}
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 14px !important;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 10px !important;
    font-family: var(--font-mono) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 22px !important;
}
/* Remover padding extra do main */
.main .block-container { padding-left: 2rem !important; padding-right: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "messages": [],           # histórico de mensagens: [{role, content, meta}]
        "agent": None,            # instância do agente LangGraph
        "agent_ready": False,     # flag de inicialização
        "agent_error": None,      # erro de inicialização, se houver
        "dados_status": {},       # status dos datasets
        "dados_ts": None,         # timestamp do último carregamento
        "ranking_df": None,       # cache do ranking de FIIs
        "ranking_ts": None,       # timestamp do ranking
        "pending_question": None, # pergunta pendente (via suggestion click)
        "total_tool_calls": 0,    # contador global de tool calls
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Carregamento do Agente ───────────────────────────────────────────────────

@st.cache_resource
def load_agent():
    """
    Inicializa o agente LangGraph uma única vez.
    @cache_resource evita reconstrução a cada rerun do Streamlit.
    """
    if not AGENT_IMPORT_OK:
        raise ImportError(f"Falha ao importar agent.py: {AGENT_IMPORT_ERROR}")
    return build_agent()


def get_agent():
    """Retorna a instância cacheada do agente, inicializando se necessário."""
    if not st.session_state.agent_ready:
        try:
            st.session_state.agent = load_agent()
            st.session_state.agent_ready = True
            st.session_state.agent_error = None
        except Exception as e:
            st.session_state.agent_ready = False
            st.session_state.agent_error = str(e)
    return st.session_state.agent


# ─── Status dos Dados ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_dados_status():
    """
    Lê o status dos datasets carregados pelo agent.py.
    Cacheado por 5 minutos.
    """
    if not AGENT_IMPORT_OK:
        return {}, datetime.datetime.now()

    try:
        dados = get_dados()
        status = {}

        if "historico_cvm" in dados and dados["historico_cvm"] is not None:
            df = dados["historico_cvm"]
            status["Histórico CVM"] = {"ok": True, "registros": len(df)}
        else:
            status["Histórico CVM"] = {"ok": False, "registros": 0}

        if "sre" in dados and dados["sre"]:
            status["SRE CVM"] = {"ok": True, "registros": len(dados["sre"])}
        else:
            status["SRE CVM"] = {"ok": False, "registros": 0}

        if "debentures" in dados and dados["debentures"]:
            status["ANBIMA Debêntures"] = {"ok": True, "registros": len(dados["debentures"])}
        else:
            status["ANBIMA Debêntures"] = {"ok": False, "registros": 0}

        if "cri_cra" in dados and dados["cri_cra"]:
            status["ANBIMA CRI/CRA"] = {"ok": True, "registros": len(dados["cri_cra"])}
        else:
            status["ANBIMA CRI/CRA"] = {"ok": False, "registros": 0}

        return status, datetime.datetime.now()
    except Exception as e:
        return {"Erro": {"ok": False, "registros": 0, "msg": str(e)}}, datetime.datetime.now()


@st.cache_data(ttl=600, show_spinner=False)
def get_ranking_cached():
    """
    Executa a tool ranking_ofertas_fii e retorna o resultado cacheado por 10 min.
    Tenta parsear a saída de texto em DataFrame estruturado.
    """
    if not AGENT_IMPORT_OK:
        return None, "Agente não disponível."

    try:
        if hasattr(ranking_ofertas_fii, "invoke"):
            resultado = ranking_ofertas_fii.invoke({})
        else:
            resultado = ranking_ofertas_fii()
        return resultado, None
    except Exception as e:
        return None, str(e)


# ─── Render: Sidebar ─────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # ── Logo / Título ──
        st.markdown("""
        <div style="padding: 8px 0 20px;">
            <div style="font-size:11px; font-family: 'IBM Plex Mono', monospace;
                        color: #4a6080; letter-spacing: 2px; text-transform: uppercase;
                        margin-bottom: 4px;">LIGAAI PLATFORM</div>
            <div style="font-size:18px; font-weight:700; color:#e8edf5; letter-spacing:-0.3px;">
                Mercado Primário
            </div>
            <div style="font-size:11px; color:#8fa3bf; margin-top:2px;">
                Análise Inteligente de Ofertas
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Datasets ──
        st.markdown("""
        <div style="font-size:10px; font-family:'IBM Plex Mono',monospace;
                    color:#4a6080; letter-spacing:1.5px; text-transform:uppercase;
                    margin-bottom:8px;">DATASETS CARREGADOS</div>
        """, unsafe_allow_html=True)

        dados_status, dados_ts = get_dados_status()

        total_registros = 0
        for nome, info in dados_status.items():
            ok = info.get("ok", False)
            regs = info.get("registros", 0)
            total_registros += regs
            icon = "●" if ok else "○"
            badge_cls = "badge-ok" if ok else "badge-warn"
            regs_str = f"{regs:,}" if regs > 0 else "—"
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1e2d47;">
                <div style="font-size:11px; color:{'#00c896' if ok else '#f0b429'};">
                    {icon} {nome}
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#8fa3bf;">
                    {regs_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Total
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:8px 0 4px;">
            <div style="font-size:10px; color:#4a6080; font-family:'IBM Plex Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.8px;">TOTAL</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                        color:#00b4d8; font-weight:600;">{total_registros:,}</div>
        </div>
        """, unsafe_allow_html=True)

        # Timestamp
        if dados_ts:
            st.markdown(f"""
            <div style="font-size:10px; color:#4a6080; font-family:'IBM Plex Mono',monospace;
                        margin-top:6px;">
                ⏱ {dados_ts.strftime('%d/%m/%Y %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider" style="margin:14px 0;"></div>', unsafe_allow_html=True)

        # ── Botões de Ação ──
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpar Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.total_tool_calls = 0
                st.rerun()            

        st.markdown('<div class="divider" style="margin:14px 0;"></div>', unsafe_allow_html=True)

        # ── Rodapé ──
        st.markdown("""
        <div style="margin-top:20px; font-size:10px; color:#2a3f5f;
                    font-family:'IBM Plex Mono',monospace; text-align:center; line-height:1.6;">
            LigaAI Platform v1.0<br>
            LangGraph · Groq · LLaMA 3.1<br>
            Dados: CVM · ANBIMA · BCB
        </div>
        """, unsafe_allow_html=True)


# ─── Render: Header ───────────────────────────────────────────────────────────

def render_header():
    col_title, col_badges = st.columns([3, 2])

    with col_title:
        st.markdown("""
        <div class="page-header">
            <h1 class="page-title">Liga<span>AI</span></h1>
            <div class="page-subtitle">
                Agente Inteligente de Análise · Mercado Primário Brasileiro
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Render: Sugestões Rápidas ────────────────────────────────────────────────

SUGGESTIONS = [
    {
        "label": "Melhores ofertas primárias",
        "text": "Quais são as melhores ofertas primárias disponíveis agora? Gere um ranking.",
    },
    {
        "label": "Spreads CRI/CRA IPCA",
        "text": "Compare os spreads dos CRIs e CRAs indexados ao IPCA. Quais oferecem maior prêmio?",
    },
    {
        "label": "FIIs mais descontados",
        "text": "Quais FIIs estão sendo ofertados com maior desconto em relação ao VP? Analise o ranking.",
    },
    {
        "label": "Selic e impacto em FIIs",
        "text": "Como a taxa Selic atual impacta os FIIs e o mercado de renda fixa? Contextualize com os dados macro.",
    },
    {
        "label": "Ofertas recentes CVM",
        "text": "Mostre as ofertas primárias mais recentes registradas na CVM via SRE. Analise o perfil das emissões.",
    },
]


def render_suggestions():
    """Renderiza os 5 cards de sugestões rápidas."""
    st.markdown("""
    <div style="font-size:10px; font-family:'IBM Plex Mono',monospace;
                color:#4a6080; letter-spacing:1.5px; text-transform:uppercase;
                margin-bottom:10px;">SUGESTÕES RÁPIDAS</div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for i, s in enumerate(SUGGESTIONS):
        with cols[i]:
            # Botão com estilo de card
            st.markdown(f"""
            <div style="background:#131c2e; border:1px solid #1e2d47; border-radius:6px;
                        padding:12px; margin-bottom:4px; min-height:80px;">
                <div style="font-size:15px; color:#8fa3bf; line-height:1.4;">{s['label']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(
                "→ Perguntar",
                key=f"sug_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_question = s["text"]
                st.rerun()


# ─── Render: Ranking de FIIs ─────────────────────────────────────────────────

def _parse_ranking_text(data):

    try:

        # se já vier dict
        if isinstance(data, dict):

            if "ranking" in data:
                return pd.DataFrame(data["ranking"])

            return pd.DataFrame(data)

        # se vier string JSON
        if isinstance(data, str):

            parsed = json.loads(data)

            if "ranking" in parsed:
                return pd.DataFrame(parsed["ranking"])

            return pd.DataFrame(parsed)

    except Exception as e:
        print(e)

    return None
    
def render_ranking():
    """Renderiza o ranking de FIIs com visual de tabela Bloomberg."""
    st.markdown("""
    <div style="font-size:10px; font-family:'IBM Plex Mono',monospace;
                color:#4a6080; letter-spacing:1.5px; text-transform:uppercase;
                margin-bottom:10px;">RANKING DE ATRATIVIDADE — FIIs</div>
    """, unsafe_allow_html=True)

    col_rank, col_btn = st.columns([5, 1])
    with col_btn:
        refresh_rank = st.button("↻", key="refresh_ranking", help="Atualizar ranking")
        if refresh_rank:
            get_ranking_cached.clear()

    ranking_raw, ranking_err = get_ranking_cached()

    if ranking_err:
        st.markdown(f"""
        <div class="error-box">
            <strong>Erro ao carregar ranking:</strong> {ranking_err}
        </div>
        """, unsafe_allow_html=True)
        return

    if not ranking_raw:
        st.markdown("""
        <div style="background:#131c2e; border:1px solid #1e2d47; border-radius:6px;
                    padding:16px; color:#8fa3bf; font-size:13px; text-align:center;">
            Dados de SRE não disponíveis.<br>
            <small style="color:#4a6080;">Rode: python src/data_ingestion/cvm_sre.py</small>
        </div>
        """, unsafe_allow_html=True)
        return

    # Tenta parsear tabela estruturada
    df_rank = _parse_ranking_text(ranking_raw)

    if df_rank is not None and not df_rank.empty:
        # Seleciona top 5
        df_show = df_rank.head(5).copy()

        # Estilização do DataFrame
        def color_score(val):
            try:
                v = float(val)
                if v >= 5:
                    return "color: #00c896; font-weight: 600;"
                elif v >= 2:
                    return "color: #00b4d8;"
                elif v >= 0:
                    return "color: #f0b429;"
                else:
                    return "color: #e53e3e;"
            except (ValueError, TypeError):
                return ""

        def color_pvp(val):
            try:
                v = float(val)
                if v < 0.95:
                    return "color: #00c896; font-weight: 600;"
                elif v < 1.00:
                    return "color: #00b4d8;"
                elif v < 1.05:
                    return "color: #f0b429;"
                else:
                    return "color: #e53e3e;"
            except (ValueError, TypeError):
                return ""

        styled = df_show.style.map(
            color_score,
            subset=[c for c in df_show.columns if "score" in c.lower()],
        ).map(
            color_pvp,
            subset=[c for c in df_show.columns if "p_vp" in c.lower() or "pvp" in c.lower()],
        ).set_properties(**{
            "background-color": "#131c2e",
            "color": "#e8edf5",
            "border-color": "#1e2d47",
            "font-family": "'IBM Plex Mono', monospace",
            "font-size": "12px",
        }).set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a2540"),
                ("color", "#8fa3bf"),
                ("font-size", "10px"),
                ("font-family", "'IBM Plex Mono', monospace"),
                ("text-transform", "uppercase"),
                ("border-color", "#1e2d47"),
            ]},
        ])

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            background-color: #131c2e !important;
            border: 1px solid #1e2d47 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] div {
            color: #d6e2f0 !important;
        }

        [data-testid="stDataFrame"] thead tr th {
            background-color: #1a2540 !important;
            color: #8fb3d9 !important;
            font-size: 11px !important;
            border-bottom: 1px solid #2a3f5f !important;
        }

        [data-testid="stDataFrame"] tbody tr {
            background-color: #131c2e !important;
        }

        [data-testid="stDataFrame"] tbody tr:hover {
            background-color: #1b2942 !important;
        }

        [data-testid="stDataFrame"] tbody td {
            border-color: #1e2d47 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=260
        )

    else:
        # Fallback: mostra o texto raw em expander + preview formatado
        with st.expander("Ver resultado bruto do ranking", expanded=True):
            # Extrai apenas o topo (antes dos insights)
            preview_lines = []
            for line in ranking_raw.split("\n")[:25]:
                preview_lines.append(line)

            st.markdown(f"""
            <div style="background:#0a0e17; border:1px solid #1e2d47; border-radius:6px;
                        padding:14px; font-family:'IBM Plex Mono',monospace; font-size:11px;
                        color:#8fa3bf; white-space:pre; overflow-x:auto; line-height:1.6;">
{chr(10).join(preview_lines)}
            </div>
            """, unsafe_allow_html=True)

    # Mostra insight extraído
    if ranking_raw and "MELHOR OFERTA" in ranking_raw:
        insight_start = ranking_raw.find("MELHOR OFERTA")
        insight_text = ranking_raw[insight_start:].strip()
        with st.expander("Insight do Agente", expanded=False):
            st.markdown(f"""
            <div style="font-size:12px; color:#8fa3bf; line-height:1.7;
                        font-family:'IBM Plex Sans',sans-serif; padding:4px;">
                {insight_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)


# ─── Processar Pergunta ───────────────────────────────────────────────────────

def processar_pergunta(pergunta: str):
    """
    Envia a pergunta ao agente LangGraph com streaming.
    Captura tool calls e resposta final, atualizando o histórico.
    """
    agent = get_agent()
    if agent is None:
        return

    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({
        "role": "user",
        "content": pergunta,
        "ts": datetime.datetime.now().isoformat(),
    })

    # Containers de streaming
    tool_calls_info = []
    resposta_final = ""
    tool_call_count = 0
    inicio = time.time()
    erro = None

    # Área de streaming na interface
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            # ── Streaming do LangGraph ──
            for step in agent.stream(
                {"messages": [{"role": "user", "content": pergunta}]}
            ):
                # ── Passo de tool call ──
                if "tools" in step:
                    tool_messages = step["tools"].get("messages", [])

                    for msg in tool_messages:
                        tool_call_count += 1

                        tool_name = getattr(msg, "name", "tool")

                        content = getattr(msg, "content", "")

                        if isinstance(content, list):
                            content = str(content)

                        preview = str(content)[:150].replace("\n", " ")

                        tool_calls_info.append({
                            "name": tool_name,
                            "preview": str(preview),
                        })

                # ── Passo de resposta do agente ──
                if "agent" in step:
                    last = step["agent"]["messages"][-1]
                    content = getattr(last, "content", "")

                    if isinstance(content, list):
                        content = "\n".join(
                            str(x.get("text", x)) if isinstance(x, dict) else str(x)
                            for x in content
                        )

                    resposta_final = str(content)
                    response_placeholder.markdown(resposta_final)

        except Exception as e:
            erro = str(e)
            tb = traceback.format_exc()
            resposta_final = f"Erro ao processar: {erro}"
            response_placeholder.markdown(f"""
            <div class="error-box">
                <strong>⚠ Erro durante processamento:</strong><br>
                {erro}<br>
                <small style="color:#4a6080;">{tb[:200]}...</small>
            </div>
            """, unsafe_allow_html=True)

        finally:
            # Limpa o status spinner
            status_placeholder.empty()

        # ── Meta-informações da resposta ──
        elapsed = time.time() - inicio
        st.session_state.total_tool_calls += tool_call_count

        if tool_calls_info:
            with st.expander(
                f"🔧 {tool_call_count} tool call{'s' if tool_call_count > 1 else ''} executada{'s' if tool_call_count > 1 else ''} · {elapsed:.1f}s",
                expanded=False,
            ):
                for tc in tool_calls_info:
                    st.markdown(f"""
                    <div style="padding:6px 0; border-bottom:1px solid #1e2d47;">
                        <div class="tool-badge" style="margin-bottom:4px;">⚙ {tc['name']}</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                                    color:#8fa3bf; padding-left:4px;">{tc['preview']}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Salva no histórico
        st.session_state.messages.append({
            "role": "agent",
            "content": resposta_final,
            "ts": datetime.datetime.now().isoformat(),
            "tool_calls": tool_calls_info,
            "tool_count": tool_call_count,
            "elapsed": round(elapsed, 2),
            "error": erro,
        })


# ─── Render: Chat ─────────────────────────────────────────────────────────────

def render_chat():
    """Renderiza o histórico completo do chat + input."""

    # ── Inicializa o agente na primeira execução ──
    if not st.session_state.agent_ready and AGENT_IMPORT_OK:
        with st.spinner("Inicializando agente LangGraph..."):
            get_agent()

    # ── Erro de importação ──
    if not AGENT_IMPORT_OK:
        st.markdown(f"""
        <div class="error-box">
            <strong>⚠ Falha ao importar agent.py</strong><br>
            {AGENT_IMPORT_ERROR}<br><br>
            Verifique se o arquivo está em <code>src/agent/agent.py</code>
            e se todas as dependências estão instaladas.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Histórico de mensagens ──
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:#4a6080;">
            <div style="font-size:32px; margin-bottom:12px;"></div>
            <div style="font-size:15px; color:#8fa3bf; font-weight:500;">
                Pronto para análise
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Renderiza cada mensagem
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
                    ts = msg.get("ts", "")
                    if ts:
                        try:
                            dt = datetime.datetime.fromisoformat(ts)
                            st.markdown(f"""
                            <div class="msg-meta">{dt.strftime('%H:%M:%S')}</div>
                            """, unsafe_allow_html=True)
                        except Exception:
                            pass

            elif msg["role"] == "agent":
                with st.chat_message("assistant"):
                    if msg.get("error"):
                        st.markdown(f"""
                        <div class="error-box">{msg['content']}</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])

                    # Meta-info
                    meta_parts = []
                    ts = msg.get("ts", "")
                    if ts:
                        try:
                            dt = datetime.datetime.fromisoformat(ts)
                            meta_parts.append(dt.strftime('%H:%M:%S'))
                        except Exception:
                            pass

                    elapsed = msg.get("elapsed")
                    if elapsed:
                        meta_parts.append(f"⏱ {elapsed}s")

                    tc = msg.get("tool_count", 0)
                    if tc:
                        meta_parts.append(f"⚙ {tc} tool call{'s' if tc > 1 else ''}")

                    if meta_parts:
                        st.markdown(f"""
                        <div class="msg-meta">{' · '.join(meta_parts)}</div>
                        """, unsafe_allow_html=True)

                    # Tool calls expandíveis
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        with st.expander(f"🔧 Ver tool calls ({len(tool_calls)})", expanded=False):
                            for tc_info in tool_calls:
                                st.markdown(f"""
                                <div style="padding:5px 0; border-bottom:1px solid #1e2d47;">
                                    <div class="tool-badge">⚙ {tc_info['name']}</div>
                                    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                                                color:#8fa3bf; margin-top:3px;">{tc_info['preview']}...</div>
                                </div>
                                """, unsafe_allow_html=True)

    # ── Auto-scroll ──
    st.markdown("""
    <div id="chat-bottom"></div>
    <script>
        const el = document.getElementById('chat-bottom');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
    </script>
    """, unsafe_allow_html=True)

    # ── Input ──
    st.markdown('<div class="input-area">', unsafe_allow_html=True)

    col_input, col_send = st.columns([9, 1])

    with col_input:
        pergunta = st.text_input(
            label="",
            placeholder="Faça sua análise... Ex: Quais FIIs com maior desconto P/VP foram emitidos em 2024?",
            key="chat_input",
            label_visibility="collapsed",
        )

    with col_send:
        enviar = st.button("→ Enviar", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Processa input ──
    # Prioridade: pergunta pendente (via suggestion) > input manual
    pergunta_final = None

    if st.session_state.pending_question:
        pergunta_final = st.session_state.pending_question
        st.session_state.pending_question = None

    elif enviar and pergunta and pergunta.strip():
        pergunta_final = pergunta.strip()

    if pergunta_final:
        if not st.session_state.agent_ready:
            st.warning("Agente ainda inicializando. Tente novamente em alguns instantes.")
        else:
            processar_pergunta(pergunta_final)

# ─── Exportar Conversa ────────────────────────────────────────────────────────

def _exportar_conversa() -> str:
    """Gera texto formatado da conversa para download."""
    lines = [
        "=" * 70,
        "LigaAI — Conversa Exportada",
        f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    for msg in st.session_state.messages:
        role = "VOCÊ" if msg["role"] == "user" else "LIGAAI"
        ts = msg.get("ts", "")
        if ts:
            try:
                dt = datetime.datetime.fromisoformat(ts)
                ts_str = dt.strftime("[%H:%M:%S]")
            except Exception:
                ts_str = ""
        else:
            ts_str = ""

        lines.append(f"{role} {ts_str}")
        lines.append("-" * 40)
        lines.append(msg["content"])

        if msg["role"] == "agent":
            tc = msg.get("tool_count", 0)
            elapsed = msg.get("elapsed", 0)
            if tc or elapsed:
                lines.append(f"[{tc} tool calls · {elapsed}s]")

        lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # Inicializa o agente imediatamente (não espera primeira pergunta)
    if not st.session_state.agent_ready and AGENT_IMPORT_OK:
        try:
            st.session_state.agent = load_agent()
            st.session_state.agent_ready = True
        except Exception as e:
            st.session_state.agent_error = str(e)

    # ── Layout ──
    render_sidebar()

    # Área principal
    render_header()

    # ── Métricas rápidas no topo ──
    if AGENT_IMPORT_OK:
        dados_status, _ = get_dados_status()
        total_ok = sum(1 for v in dados_status.values() if v.get("ok"))
        total_ds = len(dados_status)
        total_regs = sum(v.get("registros", 0) for v in dados_status.values())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Datasets Online",
                f"{total_ok}/{total_ds}",
                delta="LangGraph ReAct" if st.session_state.agent_ready else "Offline",
                delta_color="normal" if st.session_state.agent_ready else "inverse",
            )
        with col2:
            st.metric(
                "Total Registros",
                f"{total_regs:,}" if total_regs > 0 else "—",
                delta="CVM + ANBIMA",
            )
        with col3:
            msgs = st.session_state.messages
            st.metric(
                "Mensagens na Sessão",
                len([m for m in msgs if m["role"] == "user"]),
                delta="perguntas realizadas",
            )
        with col4:
            st.metric(
                "Tool Calls Totais",
                st.session_state.total_tool_calls,
                delta="execuções de ferramentas",
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Sugestões rápidas ──
    render_suggestions()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── CHAT ─────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-family:'IBM Plex Mono',monospace;
                color:#4a6080; letter-spacing:1.5px; text-transform:uppercase;
                margin-bottom:10px;">
    CHAT — ANÁLISE INTELIGENTE
    </div>
    """, unsafe_allow_html=True)

    render_chat()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RANKING ──────────────────────────
    render_ranking()


if __name__ == "__main__":
    main()