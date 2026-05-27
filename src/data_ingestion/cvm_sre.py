#!/usr/bin/env python3
"""
Coletor de Ofertas Primárias de FIIs — Portal SRE da CVM
=========================================================

O QUE É O SRE?
--------------
O SRE (Sistema de Registro de Emissões) é o portal oficial da CVM onde
toda oferta pública brasileira é registrada por obrigação legal.
FIIs, CRIs, CRAs, Debêntures — tudo passa por aqui.

Link: https://web.cvm.gov.br/sre-publico-cvm/#/consulta-oferta-publica

O TRUQUE TÉCNICO
----------------
O site parece normal, mas por dentro é um AngularJS consumindo uma
API REST que devolve JSON. Isso significa que podemos chamar essa API
diretamente, sem precisar abrir um navegador.

É como se o site tivesse uma porta dos fundos que entrega os dados
já estruturados — só precisamos saber bater nessa porta.

FLUXO DA COLETA
---------------
  [1] Busca paginada (POST)
       → recebe lista de ofertas com idRequerimento

  [2] Detalhes de cada oferta (GET)
       → emissor, ativo, indexador, prazo, documentos

  [3] Participantes (GET)
       → quem coordenou (BTG? XP? Itaú?)

  [4] Info da oferta (GET)
       → taxa final, bookbuilding, demanda

  [5] Salva JSON + CSV em data/cvm_sre/

USO
---
    python src/data_ingestion/cvm_sre.py

    # Só FIIs dos últimos 60 dias:
    python src/data_ingestion/cvm_sre.py --dias 60

    # Ver estrutura da API antes de coletar:
    python src/data_ingestion/cvm_sre.py --diagnostico
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────

# __file__ é o caminho deste script
# .parent sobe uma pasta (data_ingestion/)
# .parent novamente sobe para src/
# .parent novamente chega na raiz do projeto (meu_agente_fii/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Onde os dados coletados serão salvos
DATA_DIR = PROJECT_ROOT / "data" / "cvm_sre"

# ─── Configuração da API ──────────────────────────────────────────────────────

# Essa é a URL base do portal SRE.
# Todos os endpoints começam com ela.
BASE_URL = "https://web.cvm.gov.br/sre-publico-cvm"

# Tempo de espera entre requisições (segundos).
# Importante para não sobrecarregar o servidor da CVM.
DELAY = 0.6

# Código do tipo de ativo FII no sistema da CVM.
# Descoberto inspecionando as chamadas do frontend no DevTools.
TIPO_ATIVO_FII = "FII"


# ─── BLOCO 1: Sessão HTTP ─────────────────────────────────────────────────────
#
# Por que usar Session em vez de requests.get() direto?
#
#   requests.get()  → abre uma nova conexão TCP a cada chamada (lento)
#   Session         → reutiliza a mesma conexão (mais rápido)
#                   → mantém cookies automaticamente
#                   → aplica os headers em todas as requisições
#
# Por que esses headers específicos?
#
#   Content-Type  → avisa que estamos mandando JSON no corpo da requisição
#   Accept        → pede que o servidor responda em JSON
#   Origin/Referer → o servidor verifica de onde vem a chamada.
#                    Sem isso, pode retornar 403 (bloqueado).
#                    Estamos dizendo: "venho do próprio site da CVM"
#   User-Agent    → identifica o "navegador". Servidores bloqueiam
#                    requisições sem User-Agent ou com User-Agent de bot.

def criar_sessao() -> requests.Session:
    s = requests.Session()

    s.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": "https://web.cvm.gov.br",
        "Referer": "https://web.cvm.gov.br/sre-publico-cvm/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest"
    })

    return s


# ─── BLOCO 2: Os 4 endpoints ──────────────────────────────────────────────────
#
# Cada função abaixo representa UMA chamada à API da CVM.
# Juntas, elas constroem o perfil completo de cada oferta.

def buscar_lista(sessao, pagina=1, tamanho=20, data_inicio=None, data_fim=None) -> dict:
    """
    ENDPOINT 1 — Lista de ofertas (paginada)
    POST /rest/sitePublico/pesquisar/detalhado

    O que retorna:
      - totalRegistros: total de ofertas que existem com esses filtros
      - lista: array de ofertas, cada uma com seu idRequerimento

    O idRequerimento é a CHAVE de tudo — com ele acessamos os outros endpoints.

    Por que POST e não GET?
    → Porque estamos mandando filtros no corpo da requisição (payload JSON),
      não na URL. GET manda parâmetros na URL (?pagina=1&tipo=FII),
      POST manda no corpo. O frontend da CVM usa POST para buscas filtradas.
    """
    payload = {
    "palavraChave": "",
    "tipoOferta": "",
    "categoriaEmissor": "",
    "situacaoOferta": "",
    "dataInicial": data_inicio,
    "dataFinal": data_fim,
    "pagina": pagina,
    "tamanhoPagina": tamanho
    }
    if data_inicio:
        payload["dataInicio"] = data_inicio  # formato: "YYYY-MM-DD"
    if data_fim:
        payload["dataFim"] = data_fim

    resp = sessao.post(
        f"{BASE_URL}/rest/sitePublico/pesquisar/detalhado",
        data=json.dumps(payload),
        timeout=30,
    )
    
    
    print("\nPAYLOAD:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\nSTATUS:")
    print(resp.status_code)

    print("\nRESPOSTA:")
    print(resp.text[:5000])

    if resp.status_code != 200:
        return {} # lança exceção se status for 4xx ou 5xx
    return resp.json()


def buscar_detalhes(sessao, id_req: int) -> dict:
    """
    ENDPOINT 2 — Detalhes completos de uma oferta
    GET /rest/sitePublico/pesquisar/requerimento/{id}

    O que retorna:
      - Dados do emissor (nome, CNPJ, setor)
      - Características do ativo (indexador, prazo, garantias, série)
      - Lista de documentos com UUID para download do PDF
      - Datas (registro, início, encerramento)
    """
    resp = sessao.get(
        f"{BASE_URL}/rest/sitePublico/pesquisar/requerimento/{id_req}",
        timeout=30,
    )
    if resp.status_code != 200:
        return {}
    dados = resp.json()

    if isinstance(dados, list):
        return {}

    return dados


def buscar_participantes(sessao, id_req: int) -> dict:
    """
    ENDPOINT 3 — Participantes da operação
    GET /rest/sitePublico/pesquisar/participantes/{id}

    O que retorna:
      - coordenadorLider: quem liderou (BTG? XP? Itaú BBA?)
      - coordenadores: todos os coordenadores
      - distribuidores: quem vendeu para os investidores

    Por que isso importa para o projeto?
    → Queremos comparar ofertas por instituição.
      Esse endpoint diz exatamente quem coordenou cada emissão de FII.
    """
    resp = sessao.get(
        f"{BASE_URL}/rest/sitePublico/pesquisar/participantes/{id_req}",
        timeout=30,
    )
    if resp.status_code != 200:
        return {}
    dados = resp.json()

    if isinstance(dados, list):

        # às vezes vem lista com 1 item
        if len(dados) > 0 and isinstance(dados[0], dict):
            return dados[0]

        return {}

    return dados


def buscar_info_oferta(sessao, id_req: int) -> dict:
    """
    ENDPOINT 4 — Informações operacionais da oferta
    GET /rest/sitePublico/pesquisar/infOferta/{id}

    O que retorna (quando disponível):
      - taxaFinal: a taxa de remuneração definida no bookbuilding
      - demandaTotal: quanto os investidores pediram vs. ofertado
      - bookbuilding: se houve processo de bookbuilding
      - publicoAlvo: investidores profissionais, qualificados, etc.

    ⚠️ Atenção: nem toda oferta preenche esses campos.
       Ofertas de rito automático (ICVM 476) costumam ter taxaFinal vazia.
       Nesses casos, a taxa está só no PDF do prospecto.
    """
    resp = sessao.get(
        f"{BASE_URL}/rest/sitePublico/pesquisar/infOferta/{id_req}",
        timeout=30,
    )
    if resp.status_code != 200:
        return {}
    dados = resp.json()

    if isinstance(dados, list):

        if len(dados) > 0 and isinstance(dados[0], dict):
            return dados[0]

        return {}

    return dados


# ─── BLOCO 3: Consolidação ────────────────────────────────────────────────────
#
# Cada endpoint acima retorna um pedaço dos dados.
# Essa função junta tudo em um único dicionário "achatado" (flat).
#
# Por que "achatado"?
# → Os dados chegam aninhados:
#     { "emissor": { "nome": "...", "cnpj": "..." }, ... }
# → Para salvar em CSV precisamos de colunas diretas:
#     { "emissor_nome": "...", "emissor_cnpj": "...", ... }
#
# .get("campo", {}).get("subcampo") é o padrão seguro:
# → Se "campo" não existir, retorna {} em vez de KeyError
# → Se "subcampo" não existir, retorna None em vez de KeyError

def consolidar(
    id_req: int,
    detalhes: dict,
    participantes,
    info
) -> dict:

    info_gerais = detalhes.get("informacoesGerais", {})

    # participante pode vir lista OU dict
    if isinstance(participantes, list) and participantes:
        participante = participantes[0]
    elif isinstance(participantes, dict):
        participante = participantes
    else:
        participante = {}

    # grupos -> series -> loteInicial -> loteBase
    grupo = detalhes.get("grupos", [{}])[0]

    serie = grupo.get("series", [{}])[0]

    lote = serie.get("loteInicial", {})

    lote_base = lote.get("loteBase", {})

    campos = lote.get("camposCadastrados", [])

    descricao = None

    if campos:
        descricao = campos[0].get("campoValor")

    return {

        # ID
        "id_requerimento": id_req,

        # REGISTRO
        "numero_processo":
            info_gerais.get("numeroProcesso"),

        "numero_registro":
            info_gerais.get("numeroRegistro"),

        "status":
            info_gerais.get("status"),

        # FUNDO
        "fundo_nome":
            descricao,

        # TIPO
        "tipo_ativo":
            info_gerais.get("nomeValorMobiliario"),

        "tipo_oferta":
            detalhes.get("tipoOfertaRequerimento"),

        # VALORES
        "valor_total":
            info_gerais.get("valorTotalInicial"),

        "quantidade_cotas":
            lote_base.get("quantidadeAtivos"),

        "preco_cota":
            lote_base.get("valorNominal"),

        "taxa_distribuicao":
            lote_base.get("custoUnitario"),

        # DATAS
        "data_registro":
            info_gerais.get("data"),

        # COORDENADOR
        "coordenador":
            participante.get("razaoSocial"),

        "coordenador_cnpj":
            participante.get("cnpjInstituicao"),

        # BOOKBUILDING
        "bookbuilding":
            detalhes.get("bookPreenchido"),

        # OFERTA
        "oferta_inicial":
            info.get("valor"),

        # META
        "coletado_em":
            datetime.now().isoformat(),
    }

# ─── BLOCO 5: Salvar resultados ───────────────────────────────────────────────
#
# Salvamos em dois formatos:
#
#   JSON → preserva a estrutura original, bom para o agente consumir
#   CSV  → fácil de abrir no Excel/pandas, bom para análise rápida
#
# O timestamp no nome garante que cada coleta gere um arquivo novo,
# sem sobrescrever dados anteriores. Útil para comparar coletas ao longo do tempo.

def salvar(ofertas: list[dict], prefixo: str = "fiis") -> Path:
    """Salva as ofertas em JSON e CSV dentro de data/cvm_sre/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = DATA_DIR / f"{prefixo}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ofertas, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON salvo: {json_path}")

    # CSV (utf-8-sig = UTF-8 com BOM, que o Excel abre sem problema)
    if ofertas:
        df = pd.DataFrame(ofertas)
        csv_path = DATA_DIR / f"{prefixo}_{ts}.csv"
        df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")
        print(f"CSV salvo:  {csv_path}")
        print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")

        # Preview das colunas mais úteis para o agente
        cols = [c for c in ["fundo_nome", "coordenador", "taxa_final",
                             "valor_total", "data_registro", "publico_alvo"]
                if c in df.columns]
        if cols:
            print(f"\nPreview:\n{df[cols].head(5).to_string(index=False)}")

    return json_path


# ─── BLOCO 6: Diagnóstico ─────────────────────────────────────────────────────
#
# Essa função é para quando você quiser ENTENDER a estrutura da API
# antes de coletar tudo. Ela faz 1 requisição e imprime o JSON cru.
#
# Use isso se:
#   - O código não estiver encontrando os campos certos
#   - Quiser saber quais campos a API realmente retorna
#   - A estrutura da API mudou e precisa atualizar o código

def diagnosticar():
    """
    Testa o endpoint funcional da CVM
    e imprime um exemplo real de resposta.
    """

    print("\n[DIAGNÓSTICO] Testando endpoint da CVM...\n")

    sessao = criar_sessao()

    id_req = 26426

    url = (
        f"{BASE_URL}/rest/sitePublico/"
        f"pesquisar/informacoesGerais/{id_req}"
    )

    try:
        resp = sessao.get(url, timeout=30)

        print("STATUS:")
        print(resp.status_code)

        print("\nRESPOSTA:")
        print(resp.text[:5000])

        if resp.status_code != 200:
            print("\nEndpoint retornou erro.")
            return

        dados = resp.json()

        print("\nCAMPOS:")
        print(list(dados.keys()))

        print("\nFUNDO:")
        print(dados.get("razaoSocialFundoAssociado"))

        print("\nTIPO:")
        print(dados.get("nomeValorMobiliario"))

    except Exception as e:
        print(f"\nErro: {e}")

   
def coletar_por_ids(id_inicial=26400, id_final=26450):
    sessao = criar_sessao()

    ofertas = []

    for id_req in range(id_inicial, id_final + 1):

        url = (
            f"{BASE_URL}/rest/sitePublico/"
            f"pesquisar/informacoesGerais/{id_req}"
        )

        try:
            resp = sessao.get(url, timeout=20)

            if resp.status_code != 200:
                continue

            dados = resp.json()

            # filtra apenas FIIs
            if dados.get("nomeValorMobiliario") != "Cotas de FII":
                continue

            # -----------------------------------------
            # busca endpoints complementares
            # -----------------------------------------

            detalhes = buscar_detalhes(sessao, id_req)

            participantes = buscar_participantes(sessao, id_req)

            info = buscar_info_oferta(sessao, id_req)

            # -----------------------------------------
            # consolida
            # -----------------------------------------

            print("\nDETALHES:")
            print(json.dumps(detalhes, indent=2, ensure_ascii=False)[:3000])

            print("\nPARTICIPANTES:")
            print(json.dumps(participantes, indent=2, ensure_ascii=False)[:3000])

            print("\nINFO:")
            print(json.dumps(info, indent=2, ensure_ascii=False)[:3000])

            registro = consolidar(
                id_req,
                detalhes,
                participantes,
                info
            )

            print(
                f"[OK] {id_req} | "
                f"{registro.get('fundo_nome')}"
            )

            ofertas.append(registro)

            time.sleep(0.3)

        except Exception as e:
            print(f"[ERRO] {id_req}: {e}")

    print(f"\nTotal coletado: {len(ofertas)}")

    salvar(ofertas, prefixo="fiis_ids")
    return ofertas

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Coleta ofertas de FIIs do SRE/CVM")

    parser.add_argument(
        "--paginas",
        type=int,
        default=3,
        help="Número de páginas"
    )

    parser.add_argument(
        "--por-pagina",
        type=int,
        default=10,
        help="Ofertas por página"
    )

    parser.add_argument(
        "--dias",
        type=int,
        default=None,
        help="Filtrar últimos N dias"
    )

    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Só dados básicos"
    )

    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Diagnóstico da API"
    )

    # NOVO
    parser.add_argument(
        "--ids",
        action="store_true",
        help="Coleta usando faixa de IDs"
    )

    args = parser.parse_args()

    # ─────────────────────────────────────
    # MODO IDS
    # ─────────────────────────────────────
    if args.ids:
        coletar_por_ids()
        return

    # ─────────────────────────────────────
    # MODO DIAGNÓSTICO
    # ─────────────────────────────────────
    if args.diagnostico:
        diagnosticar()
        return

    # ─────────────────────────────────────
    # MODO NORMAL
    # ─────────────────────────────────────
    data_inicio = data_fim = None

    if args.dias:
        data_fim = datetime.now().strftime("%Y-%m-%d")

        data_inicio = (
            datetime.now() - timedelta(days=args.dias)
        ).strftime("%Y-%m-%d")

    ofertas = coletar_por_ids(
        id_inicial=26400,
        id_final=26500
    )

    if ofertas:
        salvar(ofertas)
    else:
        print("Nenhuma oferta coletada.")


if __name__ == "__main__":
    main()