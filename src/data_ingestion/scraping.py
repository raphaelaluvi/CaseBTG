#!/usr/bin/env python3
"""
Coleta de Taxas Indicativas — ANBIMA via r.jina.ai
====================================================

O QUE É ESSE SCRIPT?
--------------------
Coleta taxas indicativas diárias de renda fixa da ANBIMA Data usando
r.jina.ai como intermediário de conversão HTML → Markdown limpo.

Fontes coletadas:
  1. ANBIMA Data — Debêntures
     URL: https://data.anbima.com.br/debentures/
     O que retorna: lista de debêntures com taxa indicativa, emissor,
     duration e vencimento. Dados atualizados diariamente.
     ✅  100% público, sem autenticação.

  2. ANBIMA Data — CRI e CRA
     URL: https://data.anbima.com.br/certificado-de-recebiveis/
     O que retorna: taxas indicativas de CRI (imobiliário) e
     CRA (agronegócio). Dados públicos atualizados diariamente.
     ✅  100% público, sem autenticação.

POR QUE r.jina.ai?
------------------
Sites modernos entregam HTML cheio de scripts, menus e CSS.
O r.jina.ai transforma qualquer URL pública em Markdown limpo,
eliminando o ruído e deixando só o conteúdo relevante.

  HTML pesado (200 KB) → r.jina.ai → Markdown limpo (5–20 KB)

Isso facilita muito o trabalho do LLM que vai extrair estruturas
de dados a partir do texto.

COMO O FLUXO FUNCIONA?
-----------------------
  1. fetch_markdown(url)   → pega o Markdown via r.jina.ai
  2. extrair_anbima(md)    → manda pro ChatGroq extrair JSON estruturado
  3. salvar(dados)         → salva em data/scraped/ como JSON + CSV

EVOLUÇÃO FUTURA
---------------
A ANBIMA tem uma API REST oficial (api.anbima.com.br) com dados
estruturados em JSON. Requer cadastro gratuito + API key.
Endpoints:
    GET api.anbima.com.br/feed/precos-indices/v1/debentures/mercado-secundario
    GET api.anbima.com.br/feed/precos-indices/v1/cri-cra/mercado-secundario
Usar a API diretamente seria mais robusto que scraping via Jina.

USO
---
    python src/data_ingestion/scraping.py

    # Só debêntures
    python src/data_ingestion/scraping.py --fonte anbima

    # Só CRI/CRA
    python src/data_ingestion/scraping.py --fonte anbima_cri_cra

    # Ver markdown bruto (debug)
    python src/data_ingestion/scraping.py --debug
"""

import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "scraped"

# ─── Configuração ─────────────────────────────────────────────────────────────

# Jina Reader: converte qualquer URL pública em Markdown limpo
# Não exige API key para uso básico (limite: ~200 req/dia sem key)
# Com API key (gratuita): https://jina.ai/ → muito mais requisições
JINA_BASE = "https://r.jina.ai"

# Headers recomendados pelo Jina para melhor extração
JINA_HEADERS = {
    "Accept": "text/plain",
    "X-Return-Format": "markdown",     # retorna Markdown (vs text/html)
    "X-Timeout": "15",                 # timeout interno do Jina em segundos
}

# Fontes de dados
FONTES = {
    "anbima": {
        "label": "ANBIMA Data — Debêntures",
        "url": "https://data.anbima.com.br/debentures/",
        "descricao": (
            "Plataforma pública da ANBIMA com taxas indicativas de debêntures. "
            "Dados atualizados diariamente. Inclui emissor, taxa, duration, indexador."
        ),
    },
    "anbima_cri_cra": {
        "label": "ANBIMA Data — CRI e CRA",
        "url": "https://data.anbima.com.br/certificado-de-recebiveis/",
        "descricao": (
            "Taxas indicativas de CRI (imobiliário) e CRA (agronegócio). "
            "Dados públicos atualizados diariamente pela ANBIMA."
        ),
    },
}

# Modelo Groq para extração
# Llama 3.1 8B é mais rápido e suficiente para extração estruturada simples
# Use 70B se a extração estiver imprecisa
LLM_MODEL = "llama-3.1-8b-instant"


# ─── BLOCO 1: Fetch via r.jina.ai ────────────────────────────────────────────
#
# O Jina Reader funciona assim:
#   GET https://r.jina.ai/{URL_ALVO}
#
# Internamente ele abre um browser headless, espera o JS carregar,
# e devolve o conteúdo em Markdown limpo.
#
# Vantagem sobre requests direto:
# → Executa JavaScript (SPA/React/Angular funcionam)
# → Remove ads, menus, rodapés automaticamente
# → Retorna só o conteúdo principal da página

def fetch_markdown(url: str, debug: bool = False) -> str:
    """
    Converte uma URL pública em Markdown limpo via r.jina.ai.

    Parâmetros:
        url   — URL da página a ser convertida
        debug — se True, imprime os primeiros 500 chars do markdown

    Retorna:
        String com o conteúdo da página em Markdown
    """
    jina_url = f"{JINA_BASE}/{url}"
    print(f"  → Buscando via r.jina.ai: {url}")

    try:
        resp = requests.get(jina_url, headers=JINA_HEADERS, timeout=30)
        resp.raise_for_status()
        markdown = resp.text

        # Jina às vezes retorna uma página de erro disfarçada como 200
        if len(markdown) < 200 or "error" in markdown.lower()[:100]:
            print(f"  ⚠️  Resposta suspeita ({len(markdown)} chars)")
            return ""

        print(f"  ✓ Markdown recebido: {len(markdown):,} chars")

        if debug:
            print(f"\n{'─'*60}")
            print("MARKDOWN (primeiros 500 chars):")
            print(markdown[:500])
            print(f"{'─'*60}\n")

        return markdown

    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout (30s). Site pode estar lento ou bloqueando bots.")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP Error: {e}")
        return ""
    except Exception as e:
        print(f"  ✗ Erro inesperado: {e}")
        return ""


# ─── BLOCO 2: Extração estruturada com LLM ───────────────────────────────────
#
# Por que usar LLM aqui em vez de BeautifulSoup?
# → O Jina já limpou o HTML. O que sobrou é texto semiestruturado.
# → Um parser tradicional precisaria de XPath/CSS específico por site.
#   Se o site mudar o layout, quebra.
# → O LLM entende o contexto (o que é taxa, o que é nome, o que é prazo)
#   e extrai corretamente mesmo com variações de formato.
#
# Estratégia:
# → Mandamos o markdown + instrução de extrair JSON
# → Usamos temperature=0 para maximizar determinismo
# → Pedimos apenas os campos que realmente precisamos

def build_llm() -> ChatGroq:
    """Inicializa o ChatGroq com configurações para extração estruturada."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,       # extração precisa, sem criatividade
        max_tokens=2000,     # suficiente para um JSON com ~20 ativos
    )


def extrair_anbima(markdown: str, llm: ChatGroq) -> list[dict]:
    """
    Extrai debêntures/CRI/CRA da página da ANBIMA Data.

    Retorna lista de dicts com:
        codigo        — código do ativo (ex: VALE14, CSNA11)
        emissor       — nome do emissor
        indexador     — CDI, IPCA, IGPM, prefixado
        taxa          — taxa indicativa (spread sobre indexador ou taxa bruta)
        duration      — duration em anos quando disponível
        vencimento    — data de vencimento quando disponível
        tipo          — Debenture, CRI, CRA
    """
    if not markdown:
        return []

    # Guarda anti-alucinação: se o markdown é muito curto, a página
    # provavelmente carregou vazia (SPA com JS pesado). Não chamar o LLM.
    # Threshold: menos de 1000 chars = conteúdo insuficiente para extração real.
    if len(markdown) < 1000:
        print(f"  ⚠️  Markdown muito curto ({len(markdown)} chars) — página pode ter carregado vazia.")
        print(f"     A ANBIMA Data usa JavaScript para renderizar os dados.")
        print(f"     Retornando lista vazia (sem alucinação).")
        return []

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é um extrator de dados do mercado de capitais brasileiro. "
            "Extraia APENAS os ativos de renda fixa que estão EXPLICITAMENTE listados no texto abaixo. "
            "NUNCA invente, complete ou suponha dados que não estejam visíveis no texto. "
            "Se o texto não contiver uma tabela ou lista de ativos com códigos reais, "
            "retorne uma lista vazia: []. "
            "Retorne APENAS JSON válido sem markdown ou explicações. "
            "Formato: lista de objetos com as chaves: "
            "codigo (string), emissor (string), indexador (string), "
            "taxa (string, ex: 'CDI + 1,85%'), duration (string ou null), "
            "vencimento (string ou null), tipo (Debenture/CRI/CRA). "
            "Use null para campos ausentes. Nunca use valores fictícios ou exemplos."
        )),
        ("human", (
            "Texto da ANBIMA Data:\n\n"
            "{markdown}\n\n"
            "Extraia apenas os ativos listados com dados reais no texto acima."
        )),
    ])

    chain = prompt | llm
    resultado = chain.invoke({"markdown": markdown[:8000]})

    return _parse_json_response(resultado.content, esperado="lista")


def _parse_json_response(content: str, esperado: str = "lista") -> list[dict]:
    """
    Faz parse seguro do JSON retornado pelo LLM.

    O LLM às vezes adiciona markdown (```json ... ```) ou texto extra.
    Essa função remove o ruído e faz o parse.
    """
    content = content.strip()

    # Remove blocos markdown se presentes
    if content.startswith("```"):
        linhas = content.split("\n")
        content = "\n".join(linhas[1:-1] if linhas[-1] == "```" else linhas[1:])
        content = content.strip()

    try:
        data = json.loads(content)

        if esperado == "lista":
            if isinstance(data, list):
                return data
            # Às vezes o LLM encapsula a lista em um objeto
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
            return []

        return data

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Erro no parse do JSON: {e}")
        print(f"  Conteúdo recebido: {content[:200]}")
        return []


# ─── BLOCO 3: Enriquecimento com metadados ───────────────────────────────────
#
# Após extrair os dados brutos, adicionamos metadados úteis:
# - timestamp da coleta (para histórico)
# - fonte (para rastreabilidade no agente)
# - data de referência (para comparações temporais)

def enriquecer(dados: list[dict], fonte: str) -> list[dict]:
    """Adiciona metadados de rastreabilidade a cada registro."""
    agora = datetime.now().isoformat()
    data_ref = datetime.now().strftime("%Y-%m-%d")

    return [
        {
            **item,
            "_fonte": fonte,
            "_coletado_em": agora,
            "_data_referencia": data_ref,
        }
        for item in dados
    ]


# ─── BLOCO 4: Salvar ─────────────────────────────────────────────────────────

def salvar(dados: list[dict], nome: str) -> tuple[Path, Path]:
    """
    Salva os dados em JSON e CSV dentro de data/scraped/.

    Por que JSON E CSV?
    → JSON: preserva tipos (bool, null, nested objects) — melhor para o agente
    → CSV: fácil de abrir no Excel/Pandas — melhor para análise ad hoc
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = DATA_DIR / f"{nome}_{ts}.json"
    csv_path = DATA_DIR / f"{nome}_{ts}.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    # CSV (normaliza dicts aninhados)
    df = pd.json_normalize(dados)
    df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")

    print(f"\n  JSON salvo: {json_path}")
    print(f"  CSV salvo:  {csv_path}")
    print(f"  Registros:  {len(dados)}")

    return json_path, csv_path


# ─── BLOCO 5: Resumo ─────────────────────────────────────────────────────────

def imprimir_resumo(dados: list[dict], label: str):
    """Imprime um resumo dos dados coletados no terminal."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if not dados:
        print("  Nenhum dado extraído.")
        return

    print(f"  Total de registros: {len(dados)}")
    print(f"\n  Primeiros {min(5, len(dados))} registros:")

    for i, item in enumerate(dados[:5]):
        # Remove metadados internos para exibição limpa
        exibir = {k: v for k, v in item.items() if not k.startswith("_")}
        print(f"\n  [{i+1}] {json.dumps(exibir, ensure_ascii=False, indent=6)}")


# ─── BLOCO 6: Pipeline por fonte ─────────────────────────────────────────────

def coletar_anbima(fonte_key: str, llm: ChatGroq, debug: bool = False) -> list[dict]:
    """
    Pipeline completo para as fontes da ANBIMA.

    ANBIMA Data é 100% público — não exige login.
    Publica taxas indicativas diárias de debêntures, CRI e CRA.

    O que coletamos:
    → data.anbima.com.br/debentures/     — taxas de debêntures
    → data.anbima.com.br/certificado-de-recebiveis/ — taxas de CRI/CRA

    EVOLUÇÃO FUTURA:
    A ANBIMA tem uma API REST oficial (api.anbima.com.br) com dados
    estruturados em JSON. Requer cadastro gratuito + API key.
    Endpoint principal:
        GET /feed/precos-indices/v1/debentures/mercado-secundario
        GET /feed/precos-indices/v1/cri-cra/mercado-secundario
    Usar a API diretamente seria mais robusto que scraping via Jina.
    """
    config = FONTES[fonte_key]
    print(f"\n{'─'*60}")
    print(f"COLETANDO: {config['label']}")
    print(f"{'─'*60}")
    print(f"Fonte: {config['url']}")

    markdown = fetch_markdown(config["url"], debug=debug)
    if not markdown:
        return []

    print(f"\n  Extraindo ativos com LLM ({LLM_MODEL})...")
    dados = extrair_anbima(markdown, llm)

    if not dados:
        print(f"\n  ℹ️  Nenhum ativo extraído de '{config['label']}'.")
        if fonte_key == "anbima_cri_cra":
            print(f"     A página de CRI/CRA da ANBIMA é uma SPA React que carrega")
            print(f"     os dados via API interna após o JS executar.")
            print(f"     O Jina capturou só o shell da página, sem os ativos.")
            print(f"     Alternativa: usar a API pública da ANBIMA diretamente.")
            print(f"     Endpoint: GET api.anbima.com.br/feed/precos-indices/v1/cri-cra/mercado-secundario")
        return []
    dados = enriquecer(dados, fonte=fonte_key)

    imprimir_resumo(dados, config["label"])
    return dados


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coleta taxas indicativas da ANBIMA via r.jina.ai"
    )
    parser.add_argument(
        "--fonte",
        choices=["anbima", "anbima_cri_cra", "todas"],
        default="todas",
        help="Qual fonte coletar (padrão: todas)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Imprime markdown bruto antes da extração (útil para debugar)",
    )
    parser.add_argument(
        "--sem-llm",
        action="store_true",
        help="Pula extração LLM — salva só o markdown bruto (útil sem GROQ_API_KEY)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SCRAPING — Taxas Indicativas ANBIMA via r.jina.ai")
    print("=" * 60)

    # ── Inicializa LLM (se necessário) ──
    llm = None
    if not args.sem_llm:
        try:
            llm = build_llm()
            print(f"\n  LLM: {LLM_MODEL} (Groq)")
        except Exception as e:
            print(f"\n  ⚠️  Não foi possível inicializar o LLM: {e}")
            print("     Use --sem-llm para rodar apenas o scraping sem extração.")
            return

    # ── Coleta por fonte ──
    todos_dados = {}

    fontes_para_coletar = (
        ["anbima", "anbima_cri_cra"]
        if args.fonte == "todas"
        else [args.fonte]
    )

    for fonte_key in fontes_para_coletar:
        print(f"\n[{fontes_para_coletar.index(fonte_key)+1}/{len(fontes_para_coletar)}]", end=" ")

        if args.sem_llm:
            config = FONTES[fonte_key]
            print(f"Baixando markdown de: {config['label']}")
            markdown = fetch_markdown(config["url"], debug=True)
            if markdown:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                md_path = DATA_DIR / f"{fonte_key}_raw_{ts}.md"
                md_path.write_text(markdown, encoding="utf-8")
                print(f"  Markdown salvo: {md_path}")
            continue

        dados = coletar_anbima(fonte_key, llm, debug=args.debug)

        if dados:
            todos_dados[fonte_key] = dados
            salvar(dados, nome=fonte_key)

        # Pausa entre requisições — respeita o servidor
        if fonte_key != fontes_para_coletar[-1]:
            print("\n  Aguardando 2s antes da próxima fonte...")
            time.sleep(2)

    # ── Consolidado final ──
    if len(todos_dados) > 1:
        todos_flat = [item for lista in todos_dados.values() for item in lista]
        print(f"\n{'='*60}")
        print(f"  CONSOLIDADO: {len(todos_flat)} registros de {len(todos_dados)} fontes")
        salvar(todos_flat, nome="consolidado_renda_fixa")

    print(f"\n✓ Coleta concluída. Dados em: {DATA_DIR}")


if __name__ == "__main__":
    main()