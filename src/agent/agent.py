import json
import os
import warnings
import argparse
import unicodedata
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

# ─── De-para: CNPJ → Nome do coordenador ─────────────────────────────────────

_CNPJ_NOMES: dict[str, str] = {
    "03.751.794/0001-13": "BTG Pactual",
    "03751794000113":     "BTG Pactual",
    "02.332.886/0001-04": "Itaú BBA",
    "02332886000104":     "Itaú BBA",
    "08.769.451/0001-08": "XP Investimentos",
    "08769451000108":     "XP Investimentos",
    "02.671.743/0001-19": "Bradesco BBI",
    "02671743000119":     "Bradesco BBI",
    "78.632.767/0001-20": "Banco do Brasil",
    "78632767000120":     "Banco do Brasil",
    "17.298.092/0001-30": "Santander",
    "17298092000130":     "Santander",
    "09.304.427/0001-58": "Caixa Econômica Federal",
    "09304427000158":     "Caixa Econômica Federal",
    "89.960.090/0001-76": "Credit Suisse",
    "89960090000176":     "Credit Suisse",
    "33.264.668/0001-03": "Morgan Stanley",
    "33264668000103":     "Morgan Stanley",
    "62.232.889/0001-90": "Citibank",
    "62232889000190":     "Citibank",
}


def _cnpj_para_nome(cnpj: str) -> str:
    """
    Traduz CNPJ para nome do coordenador.
    Usa apenas cache local para evitar chamadas externas.
    """

    if not isinstance(cnpj, str) or not cnpj.strip():
        return cnpj

    cnpj_limpo = cnpj.strip()

    if cnpj_limpo in _CNPJ_NOMES:
        return _CNPJ_NOMES[cnpj_limpo]

    apenas_digitos = "".join(c for c in cnpj_limpo if c.isdigit())

    return _CNPJ_NOMES.get(apenas_digitos, cnpj_limpo)

# ─── Séries macro do BCB SGS ──────────────────────────────────────────────────
# Endpoint: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/{n}


_BCB_SERIES = {
    "432":   "Selic acumulada no mês (% a.a.)",
    "13522": "IPCA acumulado 12 meses (%)",
    "4189":  "Meta Selic - Copom (% a.a.)",
    "7326":  "IPCA variação mensal (%)",
}


# ─── Helpers de carregamento ──────────────────────────────────────────────────


def _arquivo_mais_recente(pasta: Path, padrao: str) -> Path | None:
    """Retorna o arquivo mais recente que bate com o padrão glob."""
    arquivos = sorted(pasta.glob(padrao))
    return arquivos[-1] if arquivos else None


def _carregar_csv(pasta: Path, padrao: str) -> pd.DataFrame | None:
    """Carrega o CSV mais recente que bate com o padrão."""
    path = _arquivo_mais_recente(pasta, padrao)
    if not path:
        return None
    return pd.read_csv(path, sep=";", low_memory=False)


def _carregar_json(pasta: Path, padrao: str) -> list | dict | None:
    """Carrega o JSON mais recente que bate com o padrão."""
    path = _arquivo_mais_recente(pasta, padrao)
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Carregamento inicial (uma vez ao iniciar, evita re-leitura por tool call) ─

def _inicializar_dados() -> dict:
    """
    Carrega todos os datasets na memória ao iniciar o agente.

    Por que carregar tudo de uma vez?
    → Evita latência de I/O a cada tool call durante o chat
    → Os arquivos têm no máximo alguns MBs — cabe bem na memória
    → Em produção com dados maiores, o ideal seria lazy loading ou cache
    """
    dados = {}

    # ── Histórico CVM (CSV de FIIs desde 1989) ──
    df_hist = _carregar_csv(DATA_DIR / "cvm", "fiis_historico_*.csv")
    if df_hist is not None:
        # Normaliza coluna de data para facilitar filtros por ano
        for col in df_hist.columns:
            if "DT_" in col or "DATA" in col.upper() or "Dt_" in col:
                try:
                    df_hist[col] = pd.to_datetime(df_hist[col], errors="coerce")
                except Exception:
                    pass
        dados["historico_cvm"] = df_hist
        print(f"Histórico CVM: {len(df_hist):,} registros")
    else:
        print(f"Histórico CVM não encontrado. Rode: python src/data_ingestion/cvm_csv.py")

    # ── Ofertas recentes SRE (JSON da API CVM) ──
    sre = _carregar_json(DATA_DIR / "cvm_sre", "fiis_*.json")
    if sre is not None:
        # O JSON pode ser lista direta ou dict com lista dentro
        dados["sre"] = sre if isinstance(sre, list) else list(sre.values())[0]
        print(f"SRE CVM: {len(dados['sre'])} ofertas recentes")
    else:
        print(f"SRE CVM não encontrado. Rode: python src/data_ingestion/cvm_sre.py")

    # ── ANBIMA Debêntures (JSON do scraping) ──
    # Exclui o consolidado para pegar só o arquivo específico de debêntures
    debentures = _carregar_json(DATA_DIR / "scraped", "anbima_2*.json")
    if debentures is not None:
        dados["debentures"] = debentures if isinstance(debentures, list) else []
        print(f"ANBIMA Debêntures: {len(dados['debentures'])} ativos")
    else:
        print(f"ANBIMA Debêntures não encontrado. Rode: python src/data_ingestion/scraping.py")

    # ── ANBIMA CRI/CRA (JSON do scraping) ──
    cri_cra = _carregar_json(DATA_DIR / "scraped", "anbima_cri_cra_*.json")
    if cri_cra is not None:
        dados["cri_cra"] = cri_cra if isinstance(cri_cra, list) else []
        print(f"ANBIMA CRI/CRA: {len(dados['cri_cra'])} ativos")
    else:
        print(f"ANBIMA CRI/CRA não encontrado. Rode: python src/data_ingestion/scraping.py")

    return dados


# Carrega uma única vez — variável global acessível pelas tools
_DADOS = _inicializar_dados()

def get_dados() -> dict:
    """
    Recarrega os dados dinamicamente.
    Evita ficar preso a dados antigos em memória.
    """
    return _inicializar_dados()

# ─── Insights analíticos ─────────────────────────────────────────────────────

def interpretar_spread(spread: float) -> str:
    """
    Gera uma interpretação qualitativa do spread de crédito.
    """

    if spread >= 10:
        return (
            "O spread está muito acima da média do mercado, "
            "indicando percepção elevada de risco de crédito "
            "ou baixa liquidez do ativo."
        )

    elif spread >= 7:
        return (
            "O spread está elevado em relação às debêntures comparáveis, "
            "sugerindo prêmio adicional de risco."
        )

    elif spread >= 5:
        return (
            "O spread está em patamar moderado, "
            "compatível com emissões corporativas intermediárias."
        )

    return (
        "O spread está dentro do padrão esperado para o mercado."
    )

# ─── Classificação P/VP ──────────────────────────────────────────────────────

def classificar_p_vp(p_vp: float) -> tuple[str, float]:
    """
    Classifica o valuation da oferta com base no P/VP.

    Retorna:
    - classificação textual
    - bônus/penalização no score
    """

    if p_vp <= 0.90:
        return "MUITO ATRATIVO", 5

    elif p_vp <= 0.95:
        return "ATRATIVO", 3

    elif p_vp < 1.00:
        return "LEVEMENTE ATRATIVO", 1.5

    elif p_vp == 1.00:
        return "NEUTRO", 0

    elif p_vp <= 1.05:
        return "CARO", -2

    elif p_vp <= 1.10:
        return "MUITO CARO", -4

    return "EXTREMAMENTE CARO", -6

# ─── Score de atratividade ───────────────────────────────────────────────────

def calcular_score_oferta(
    preco_emissao: float,
    vp_cota: float,
    dy_anual: float,
    spread: float,
    liquidez: float,
    media_spread: float,
    ) -> float:
    """
    Calcula um score simples de atratividade para ofertas primárias de FIIs.

    Componentes:
    - desconto sobre VP
    - dividend yield
    - liquidez
    - penalização por spread excessivo (risco)

    Quanto MAIOR o score, mais atrativa parece a oferta.
    """

    # ── 1. Desconto/Premium sobre VP ──
    desconto_vp = ((vp_cota - preco_emissao) / vp_cota) * 100

    # penaliza emissão acima do VP
    if desconto_vp < 0:
        desconto_vp *= 1.5

    # ── 2. Score de DY ──
    score_dy = dy_anual

    # ── 3. Liquidez ──
    score_liquidez = min(liquidez / 10_000_000, 5)

    # ── 4. Penalização de risco ──
    penalizacao_risco = 0

    if spread > media_spread * 1.5:
        penalizacao_risco = 3

    elif spread > media_spread * 1.2:
        penalizacao_risco = 1.5

    # ── Score final ──
    score_final = (
        (0.30 * desconto_vp)
        + (0.25 * score_dy)
        + (0.15 * score_liquidez)
        - (0.30 * penalizacao_risco)
    )

    return round(score_final, 2)

def preparar_base_ofertas() -> pd.DataFrame:
    """
    Constrói base quantitativa de ofertas primárias
    usando os dados do SRE/CVM.
    """

    sre = _DADOS.get("sre")

    if not sre:
        return pd.DataFrame()

    df = pd.DataFrame(sre)

    if df.empty:
        return pd.DataFrame()

    colunas_numericas = [
        "preco_cota",
        "valor_total",
        "quantidade_cotas",
        "taxa_final",
        "demanda_total",
    ]

    for col in colunas_numericas:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
            
    # Remove inválidos

    df = df.dropna(subset=["preco_cota"])

    if df.empty:
        return pd.DataFrame()

    # VP PROXY

    df["vp_cota"] = (
        df["preco_cota"] * 1.07
    )

    # P/VP

    df["P_VP"] = (
        df["preco_cota"]
        / df["vp_cota"]
    )

    # DY PROXY

    #
    # lógica inicial:
    # fundos maiores tendem a yields menores
    #

    df["dy"] = 10

    # Liquidez proxy

    df["liquidez"] = (
        df["valor_total"]
        .fillna(0)
    )

    return df


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def resumo_historico_fiis() -> str:
    """
    Retorna um resumo geral do histórico de ofertas de FIIs registradas na CVM
    desde 1989: total de registros, volume financeiro total, breakdown por ano
    e pelos maiores coordenadores líderes de distribuição.
    Use quando o usuário pedir uma visão geral do mercado de FIIs, quiser saber
    o histórico de emissões ou quais bancos lideraram mais ofertas ao longo do tempo.
    """
    df = preparar_base_ofertas()
    if df is None:
        return "Dados históricos da CVM não disponíveis. Rode cvm_csv.py primeiro."

    total = len(df)

    # Detecta coluna de valor
    col_valor = next((c for c in df.columns if "VL_" in c or "VALOR" in c.upper()), None)
    volume_str = ""
    if col_valor:
        vol = pd.to_numeric(df[col_valor], errors="coerce").sum()
        volume_str = f"Volume total histórico: R$ {vol/1e9:.1f} bilhões\n"

    # Detecta coluna de ano
    col_data = next((c for c in df.columns if df[c].dtype == "datetime64[ns]"), None)
    anos_str = ""
    if col_data:
        anos = (
            df[col_data]
            .dt.year
            .value_counts()
            .sort_index(ascending=False)
            .head(5)
        )
        anos_str = f"\nOfertas por ano (últimos 5):\n{anos.to_string()}\n"

    # Detecta coluna de líder e traduz CNPJs para nomes
    col_lider = next((c for c in df.columns if "LIDER" in c.upper() or "NM_LIDER" in c.upper()), None)
    lideres_str = ""
    if col_lider:
        # Traduz CNPJ → nome antes de contar
        nomes = df[col_lider].astype(str).replace(_CNPJ_NOMES)
        
        # Usa apenas mapa local (evita milhares de requests HTTP)
        nomes = (
            df[col_lider]
            .astype(str)
            .replace(_CNPJ_NOMES)
            .fillna(df[col_lider])
        )

        lideres = nomes.value_counts().head(5)
        lideres_str = f"\nTop 5 coordenadores líderes:\n{lideres.to_string()}\n"

    return (
        f"HISTÓRICO DE FIIs — CVM (desde 1989)\n"
        f"Total de registros: {total:,}\n"
        f"{volume_str}"
        f"{anos_str}"
        f"{lideres_str}"
    )


@tool
def buscar_historico_fiis(
    coordenador: str = "",
    ano_inicio: int = 0,
    ano_fim: int = 0,
    limite: int = 10
) -> str:
    """
    Busca e filtra o histórico de ofertas de FIIs da CVM com filtros opcionais.
    """

    df = _DADOS.get("historico_cvm")

    if df is None:
        return "Dados históricos da CVM não disponíveis."

    resultado = df.copy()

    def normalizar(s: str) -> str:
        return (
            unicodedata
            .normalize("NFD", str(s))
            .encode("ascii", "ignore")
            .decode()
            .lower()
        )

    # Detecta colunas importantes

    col_lider = next(
        (
            c for c in resultado.columns
            if "LIDER" in c.upper()
        ),
        None
    )

    col_data = next(
        (
            c for c in resultado.columns
            if pd.api.types.is_datetime64_any_dtype(resultado[c])
        ),
        None
    )

    # Filtro por coordenador

    if coordenador and col_lider:
        resultado = resultado[
            resultado[col_lider]
            .astype(str)
            .map(normalizar)
            .str.contains(normalizar(coordenador), na=False)
        ]

    # Filtro por ano

    if col_data:
        if ano_inicio:
            resultado = resultado[
                resultado[col_data].dt.year >= ano_inicio
            ]

        if ano_fim:
            resultado = resultado[
                resultado[col_data].dt.year <= ano_fim
            ]

    if resultado.empty:
        return "Nenhum registro encontrado."

    # Seleção de colunas relevantes

    colunas_desejadas = []

    if col_data:
        colunas_desejadas.append(col_data)

    if col_lider:
        colunas_desejadas.append(col_lider)

    colunas_desejadas += [
        c for c in resultado.columns
        if any(
            k in c.upper()
            for k in [
                "EMISSOR",
                "NOME",
                "VL_",
                "VALOR",
                "STATUS",
                "TIPO"
            ]
        )
    ]

    # remove duplicadas
    colunas = list(dict.fromkeys(colunas_desejadas))

    exibir = resultado[colunas].copy()

    # Ordenação segura

    if col_data and col_data in exibir.columns:
        exibir = exibir.sort_values(
            by=col_data,
            ascending=False
        )

    exibir = exibir.head(limite)

    # Traduz CNPJ → nome

    if col_lider and col_lider in exibir.columns:
        exibir[col_lider] = (
            exibir[col_lider]
            .astype(str)
            .replace(_CNPJ_NOMES)
        )

    return (
        f"Encontrados {len(resultado):,} registros "
        f"(exibindo {min(limite, len(resultado))}):\n\n"
        + exibir.to_string(index=False)
    )

@tool
def ofertas_recentes_sre() -> str:
    """
    Retorna as ofertas de FIIs mais recentes coletadas via API SRE da CVM.
    Dados mais atuais que o histórico CSV — inclui ofertas em andamento.
    Use quando o usuário perguntar sobre ofertas recentes, em andamento,
    ou quiser saber o que está sendo registrado agora na CVM.
    """
    sre = _DADOS.get("sre")
    if not sre:
        return "Dados SRE não disponíveis. Rode cvm_sre.py primeiro."

    # Normaliza para lista de dicts se vier em outro formato
    registros = sre if isinstance(sre, list) else []
    if not registros:
        return "Nenhuma oferta recente encontrada no SRE."

    df = pd.DataFrame(registros)
    total = len(df)

    # Mostra primeiras linhas como tabela
    exibir = df.head(5)

    return (
        f"OFERTAS RECENTES — SRE CVM\n"
        f"Total de registros: {total}\n\n"
        f"Primeiras {min(5, total)} ofertas:\n"
        + exibir.to_string(index=False)
    )


@tool
def taxas_debentures_anbima() -> str:
    """
    Analisa debêntures da ANBIMA e retorna insights estruturados.
    """

    dados = get_dados()
    debentures = dados.get("debentures")

    if not debentures:
        return "Dados de debêntures não disponíveis."

    df = pd.DataFrame(debentures)

    col_taxa = next(
        (
            c for c in df.columns
            if any(k in c.lower() for k in ["spread", "taxa", "yield"])
        ),
        None
    )

    if not col_taxa:
        return "Coluna de taxa não encontrada."

    try:
        df[col_taxa] = (
            df[col_taxa]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
            .astype(float)
        )

    except Exception:
        return "Erro ao converter taxas."

    media_spread = df[col_taxa].mean()

    maior = df.loc[df[col_taxa].idxmax()]

    if "indexador" in df.columns:
        comparativos = df[df["indexador"] == maior["indexador"]]
    else:
        comparativos = df.copy()

    media_comparavel = comparativos[col_taxa].mean()

    desvio = (
        (maior[col_taxa] - media_comparavel)
        / media_comparavel
    ) * 100

    resultado = {
        "total_ativos": len(df),
        "media_spread_mercado": round(media_spread, 2),
        "ativo_maior_spread": {
            "codigo": maior.get("codigo", "N/D"),
            "emissor": maior.get("emissor", "N/D"),
            "indexador": maior.get("indexador", "N/D"),
            "taxa": round(maior[col_taxa], 2),
            "desvio_vs_mercado": round(desvio, 2),
            "interpretacao": interpretar_spread(maior[col_taxa]),
        }
    }

    if "indexador" in df.columns:
        resultado["distribuicao_indexadores"] = (
            df["indexador"]
            .value_counts()
            .to_dict()
        )

    return json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2,
    )

@tool
def taxas_cri_cra_anbima() -> str:
    """
    Retorna as taxas indicativas de CRI (Certificados de Recebíveis Imobiliários)
    e CRA (Certificados de Recebíveis do Agronegócio) publicadas pela ANBIMA Data.
    Inclui: código, emissor, indexador, taxa, duration e vencimento.
    Use quando o usuário perguntar sobre CRI, CRA, títulos isentos de IR,
    mercado imobiliário ou agronegócio em renda fixa.
    """
    cri_cra = _DADOS.get("cri_cra")
    if not cri_cra:
        return "Dados de CRI/CRA não disponíveis. Rode scraping.py primeiro."

    df = pd.DataFrame(cri_cra)
    df = df[[c for c in df.columns if not c.startswith("_")]]

    total = len(df)
    
    # ── Comparação de spreads IPCA+ ──
    comparacao_str = ""

    col_taxa = next(
        (
            c for c in df.columns
            if any(k in c.lower() for k in ["taxa", "spread", "yield"])
        ),
        None
    )

    if col_taxa and "indexador" in df.columns:
        try:
            # Converte taxa para float
            df[col_taxa] = (
                df[col_taxa]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+)")[0]
                .astype(float)
            )

            ipca = df[
                df["indexador"]
                .astype(str)
                .str.contains("IPCA", case=False, na=False)
            ]

            if not ipca.empty:
                maiores = ipca.sort_values(col_taxa, ascending=False).head(5)

                comparacao_str = (
                    "\nMAIORES SPREADS IPCA+:\n"
                    + maiores[
                        [c for c in ["codigo", "emissor", col_taxa] if c in maiores.columns]
                    ].to_string(index=False)
                    + "\n"
                )

        except Exception:
            pass

    # Breakdown CRI vs CRA
    breakdown_str = ""
    if "tipo" in df.columns:
        breakdown = df["tipo"].value_counts()
        breakdown_str = f"\nCRI vs CRA:\n{breakdown.to_string()}\n"

    # Breakdown por indexador
    indexador_str = ""
    if "indexador" in df.columns:
        por_indexador = df["indexador"].value_counts()
        indexador_str = f"\nPor indexador:\n{por_indexador.to_string()}\n"

    return (
        f"CRI e CRA — ANBIMA Data\n"
        f"Total de ativos: {total}\n"
        f"{breakdown_str}"
        f"{indexador_str}"
        f"{comparacao_str}\n"
        f"Todos os ativos:\n"
        + df.head(5).to_string(index=False)
    )


@tool
def indicadores_macro_bcb() -> str:
    """
    Retorna os principais indicadores macroeconômicos do Brasil em tempo real,
    consultando a API pública do Banco Central (BCB SGS).
    Inclui: Taxa Selic (meta e acumulada), IPCA mensal e acumulado 12 meses.
    Use quando o usuário perguntar sobre o cenário macro, Selic, inflação, Copom,
    ou quiser contextualizar as taxas de renda fixa com o ambiente econômico atual.
    Correlacione esses dados com os spreads das debêntures/CRI/CRA quando relevante.
    """
    # BCB SGS: API REST pública, sem autenticação
    # https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/{n}
    base = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados/ultimos/1?formato=json"

    resultados = []

    for codigo, descricao in _BCB_SERIES.items():
        try:
            resp = requests.get(base.format(codigo), timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    ultimo = data[-1]
                    resultados.append(
                        f"  {descricao}: {ultimo['valor']}% (referência: {ultimo['data']})"
                    )
                else:
                    resultados.append(f"  {descricao}: sem dados")
            else:
                resultados.append(f"  {descricao}: indisponível (HTTP {resp.status_code})")
        except Exception as e:
            resultados.append(f"  {descricao}: erro ao consultar BCB ({e})")

    if not resultados:
        return "Não foi possível consultar o BCB. Verifique a conexão com a internet."

    linhas = ["INDICADORES MACROECONÔMICOS — Banco Central do Brasil (BCB SGS)"]
    linhas += resultados
    linhas += [
        "",
        "Contexto para análise de renda fixa:",
        "  • Spread real = taxa do ativo − Selic ou IPCA (dependendo do indexador)",
        "  • CRIs/CRAs indexados ao IPCA: quanto maior o IPCA, maior a rentabilidade total",
        "  • Debêntures CDI+: acompanham a Selic — spread indica prêmio de crédito do emissor",
    ]

    return "\n".join(linhas)

# ─── Ranking de ofertas ──────────────────────────────────────────────────────

@tool
def ranking_ofertas_fii() -> str:
    """
    Gera ranking quantitativo de ofertas primárias de FIIs
    usando:

    - P/VP
    - Dividend Yield
    - Liquidez
    - Spread/Risco

    Quanto maior o score,
    mais atrativa parece a oferta.
    """

    df = preparar_base_ofertas()

    if df.empty:
        return "Não há ofertas disponíveis."

    df = df.copy()

    # Colunas fixas do SRE

    col_preco = "preco_cota"
    col_vp = "vp_cota"
    col_dy = "dy"
    col_liquidez = "liquidez"
    col_nome = "fundo_nome"

    # Verificação mínima

    if not col_preco or not col_vp:
        return (
            "Não foi possível encontrar colunas "
            "de preço de emissão e VP."
        )

    # Conversão numérica

    for col in [
        col_preco,
        col_vp,
        col_dy,
        col_liquidez,
    ]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Remove inválidos

    df = df.dropna(
        subset=[col_preco, col_vp]
    )

    if df.empty:
        return "Não há dados suficientes para gerar ranking."

    # Cria P/VP

    df["P_VP"] = (
        df[col_preco]
        / df[col_vp]
    )
    
    # Placeholder de spread

    media_spread = 5

    # Score

    scores = []
    classificacoes = []

    for _, row in df.iterrows():

        preco = row[col_preco]
        vp = row[col_vp]

        dy = (
            row[col_dy]
            if col_dy and pd.notna(row[col_dy])
            else 8
        )

        liquidez = (
            row[col_liquidez]
            if col_liquidez and pd.notna(row[col_liquidez])
            else 1_000_000
        )

        p_vp = row["P_VP"]

        # Classificação valuation

        classificacao, bonus_pvp = classificar_p_vp(
            p_vp
        )

        # Proxy de risco

        spread = p_vp * 10

        # Score base

        score_base = calcular_score_oferta(
            preco_emissao=preco,
            vp_cota=vp,
            dy_anual=dy,
            liquidez=liquidez,
            spread=spread,
            media_spread=media_spread,
        )


        # Score final ajustado


        score_final = (
            score_base
            + bonus_pvp
        )

        scores.append(score_final)
        classificacoes.append(classificacao)
        
    # Salva resultado

    df["score"] = scores
    df["classificacao_pvp"] = classificacoes

    # Ordena ranking

    ranking = (
        df.sort_values(
            "score",
            ascending=False
        )
        .head(5)
    )

    # Colunas finais

    colunas = [
        c for c in [
            col_nome,
            col_preco,
            col_vp,
            "P_VP",
            "classificacao_pvp",
            col_dy,
            "score",
        ]
        if c and c in ranking.columns
    ]

    # Formatação

    ranking["P_VP"] = (
        ranking["P_VP"]
        .round(2)
    )

    ranking["score"] = (
        ranking["score"]
        .round(2)
    )

    # Insights automáticos

    melhor = ranking.iloc[0]

    insight = f"""
    MELHOR OFERTA IDENTIFICADA:

    - Ativo: {melhor[col_nome]}
    - P/VP: {melhor['P_VP']}
    - Classificação: {melhor['classificacao_pvp']}
    - Score final: {melhor['score']}

    Interpretação:
    A oferta apresenta valuation descontado
    em relação ao valor patrimonial,
    combinado com bom equilíbrio entre
    yield, liquidez e risco implícito.
    """

    return json.dumps({
        "ranking": ranking[colunas].to_dict(orient="records"),
        "melhor_oferta": {
            "nome": melhor[col_nome],
            "p_vp": float(melhor["P_VP"]),
            "score": float(melhor["score"]),
            "classificacao": melhor["classificacao_pvp"]
        }
    }, ensure_ascii=False)

@tool
def comparar_ativos(codigo_1: str, codigo_2: str) -> str:
    """
    Compara dois ativos de crédito privado.
    """

    dados = get_dados()
    debentures = dados.get("debentures")

    if not debentures:
        return "Dados não disponíveis."

    df = pd.DataFrame(debentures)

    ativo1 = df[
        df.astype(str)
        .apply(
            lambda x: x.str.contains(
                codigo_1,
                case=False,
                na=False
            )
        )
        .any(axis=1)
    ]

    ativo2 = df[
        df.astype(str)
        .apply(
            lambda x: x.str.contains(
                codigo_2,
                case=False,
                na=False
            )
        )
        .any(axis=1)
    ]

    if ativo1.empty or ativo2.empty:
        return "Não foi possível localizar ambos os ativos."

    a1 = ativo1.iloc[0]
    a2 = ativo2.iloc[0]

    return json.dumps(
        {
            "ativo_1": {
                "codigo": a1.get("codigo"),
                "emissor": a1.get("emissor"),
                "indexador": a1.get("indexador"),
                "taxa": a1.get("taxa"),
            },
            "ativo_2": {
                "codigo": a2.get("codigo"),
                "emissor": a2.get("emissor"),
                "indexador": a2.get("indexador"),
                "taxa": a2.get("taxa"),
            },
            "analise": (
                "Comparar diferença de remuneração, "
                "risco implícito e cenário macroeconômico."
            )
        },
        ensure_ascii=False,
        indent=2,
    )

# ─── Agente ───────────────────────────────────────────────────────────────────

def build_agent():
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY não encontrada. Defina no arquivo .env")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    tools = [
        resumo_historico_fiis,
        buscar_historico_fiis,
        ofertas_recentes_sre,
        taxas_debentures_anbima,
        taxas_cri_cra_anbima,
        indicadores_macro_bcb,
        ranking_ofertas_fii,
        comparar_ativos,
    ]

    system_prompt = """Você é um analista do mercado de capitais brasileiro especializado em:

    - FIIs
    - debêntures
    - CRIs
    - CRAs
    - ofertas primárias
    - spreads de crédito
    - renda fixa estruturada

    Seu objetivo é ajudar o usuário a:

    - identificar ofertas atrativas;
    - comparar ativos;
    - interpretar spreads e riscos;
    - contextualizar dados com cenário macroeconômico;
    - explicar os motivos por trás das diferenças de retorno.

    Regras importantes:

    - Use apenas as tools necessárias.
    - Evite chamadas redundantes.
    - Se uma tool já responder a pergunta, use diretamente o resultado.
    - Para rankings e atratividade de FIIs, priorize `ranking_ofertas_fii`.
    - Para cenário econômico, inflação ou Selic, use `indicadores_macro_bcb`.
    - Para emissões recentes, use `ofertas_recentes_sre`.

    Ao responder:

    - Seja objetivo e claro.
    - Sempre cite os ativos mais relevantes encontrados.
    - Explique brevemente por que o ativo parece atrativo ou arriscado.
    - Não invente dados.
    - Não ignore o output das tools.
    - Nunca substitua os dados retornados por respostas genéricas.

    Responda sempre em português do Brasil.
    """

    return create_react_agent(llm, tools, prompt=system_prompt)


# ─── Loop interativo ──────────────────────────────────────────────────────────

def chat(agent, pergunta: str):
    """Envia uma pergunta ao agente e imprime a resposta com streaming por passos."""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": pergunta}]},
        stream_mode="updates",
    ):
        # Passo de tool call — mostra qual ferramenta foi chamada e preview do resultado
        if "tools" in step:
            for msg in step["tools"]["messages"]:
                preview = msg.content[:120].replace("\n", " ")
                print(f"\n  [tool: {msg.name}] → {preview}...")

        # Passo final — imprime a resposta do agente
        if "agent" in step:
            last = step["agent"]["messages"][-1]
            if last.content:
                print(f"\nAgente: {last.content}")


def main():
    parser = argparse.ArgumentParser(description="Agente de análise de ofertas primárias")
    parser.add_argument(
        "--pergunta",
        type=str,
        default=None,
        help="Roda uma pergunta direta e encerra (sem loop interativo)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LigaAI — Case BTG")
    print("  Dados: CVM Histórico | SRE CVM | ANBIMA Debêntures | CRI/CRA")
    print("  Digite 'sair' para encerrar")
    print("=" * 60)

    agent = build_agent()

    # Modo direto: roda uma pergunta e sai
    if args.pergunta:
        print(f"\nVocê: {args.pergunta}")
        chat(agent, args.pergunta)
        return

    # Modo interativo: loop até o usuário digitar 'sair'
    while True:
        try:
            pergunta = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando agente.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando agente.")
            break

        chat(agent, pergunta)


if __name__ == "__main__":
    main()