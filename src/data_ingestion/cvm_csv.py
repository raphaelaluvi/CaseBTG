#!/usr/bin/env python3
"""
Histórico de Ofertas de FIIs — CVM Dados Abertos (CSV)
=======================================================

O QUE É ESSE SCRIPT?
--------------------
A CVM disponibiliza publicamente um arquivo ZIP com o histórico completo
de todas as ofertas públicas registradas desde 1989.

URL: https://dados.cvm.gov.br/dados/OFERTA/DISTRB/DADOS/

Esse script:
  1. Baixa o ZIP (~5 MB)
  2. Extrai os CSVs
  3. Filtra só as ofertas de FII
  4. Salva um CSV limpo em data/cvm/

POR QUE USAR ESSE SCRIPT ALÉM DO cvm_sre.py?
--------------------------------------------
O cvm_sre.py coleta ofertas RECENTES via API (últimos meses).
Este script pega o HISTÓRICO COMPLETO (desde 1989).

Juntos, eles dão ao agente:
  → Contexto histórico: como eram as ofertas de FII há 5, 10 anos?
  → Volume de dados: dezenas de milhares de registros para análise
  → Comparação temporal: a taxa de hoje está alta ou baixa historicamente?

USO
---
    python src/data_ingestion/cvm_csv.py

    # Ver resumo sem salvar
    python src/data_ingestion/cvm_csv.py --preview

    # Filtrar por ano
    python src/data_ingestion/cvm_csv.py --ano 2024
"""

import io
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "cvm"

# ─── Configuração ─────────────────────────────────────────────────────────────

# URL do ZIP de ofertas no portal de dados abertos da CVM
# Esse arquivo é atualizado periodicamente pela própria CVM
ZIP_URL = "https://dados.cvm.gov.br/dados/OFERTA/DISTRIB/DADOS/oferta_distribuicao.zip"

# Nome do arquivo CSV dentro do ZIP
CSV_NOME = "oferta_distribuicao.csv"

# Encoding do CSV da CVM (latin-1 é comum em sistemas governamentais brasileiros)
ENCODING = "latin-1"

# Separador do CSV
SEPARADOR = ";"

# Colunas que identificam FIIs no dataset
# O campo CAT_EMISSOR contém a categoria do emissor
CATEGORIA_FII = "Fundo de Investimento Imobiliário - FII"


# ─── BLOCO 1: Download do ZIP ─────────────────────────────────────────────────
#
# Por que baixar como stream?
# → O arquivo tem ~5 MB. Sem stream, o requests baixa tudo na memória antes
#   de retornar. Com stream=True, baixamos em pedaços (chunks) e mostramos
#   o progresso — melhor experiência para o usuário.
#
# Por que io.BytesIO?
# → zipfile.ZipFile espera um objeto "file-like" (que tenha .read(), .seek())
#   io.BytesIO transforma os bytes baixados em um objeto assim,
#   sem precisar salvar o ZIP em disco primeiro.

def baixar_zip(url: str = ZIP_URL) -> io.BytesIO:
    """
    Baixa o ZIP da CVM e retorna como objeto em memória.

    Por que em memória e não em disco?
    → Mais rápido e não polui a pasta do projeto com arquivos temporários.
      O ZIP é descartado após a extração.
    """
    print(f"Baixando ZIP da CVM...")
    print(f"  URL: {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    # Calcula tamanho total para mostrar progresso
    total = int(resp.headers.get("content-length", 0))
    total_mb = total / 1_000_000

    buffer = io.BytesIO()
    baixado = 0

    # Baixa em chunks de 1 MB
    for chunk in resp.iter_content(chunk_size=1_024 * 1_024):
        buffer.write(chunk)
        baixado += len(chunk)
        if total:
            pct = baixado / total * 100
            print(f"  {baixado/1_000_000:.1f} MB / {total_mb:.1f} MB ({pct:.0f}%)", end="\r")

    print(f"\n  Download concluído: {baixado/1_000_000:.1f} MB")

    # Volta o cursor para o início — necessário para o zipfile conseguir ler
    buffer.seek(0)
    return buffer


# ─── BLOCO 2: Extração e leitura do CSV ──────────────────────────────────────
#
# O ZIP contém mais de um arquivo CSV.
# Usamos zipfile para listar o conteúdo e extrair só o que precisamos.

def extrair_csv(buffer: io.BytesIO, nome_csv: str = CSV_NOME) -> pd.DataFrame:
    """
    Extrai o CSV do ZIP em memória e retorna como DataFrame.

    Por que latin-1 e não utf-8?
    → Sistemas governamentais brasileiros mais antigos usam latin-1
      (também chamado ISO-8859-1). Usar utf-8 causaria erros em
      caracteres como ã, ç, é.
    """
    with zipfile.ZipFile(buffer) as zf:
        # Lista os arquivos dentro do ZIP (útil para debug)
        arquivos = zf.namelist()
        print(f"\nArquivos no ZIP: {arquivos}")

        if nome_csv not in arquivos:
            # Tenta achar um arquivo parecido
            candidatos = [a for a in arquivos if a.endswith(".csv")]
            if candidatos:
                nome_csv = candidatos[0]
                print(f"  Usando: {nome_csv}")
            else:
                raise FileNotFoundError(f"Nenhum CSV encontrado no ZIP. Arquivos: {arquivos}")

        with zf.open(nome_csv) as f:
            df = pd.read_csv(
                f,
                sep=SEPARADOR,
                encoding=ENCODING,
                low_memory=False,   # evita warning de tipos mistos em colunas grandes
            )

    print(f"  CSV carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return df


# ─── BLOCO 3: Filtrar FIIs ────────────────────────────────────────────────────
#
# O CSV tem ofertas de todos os tipos: ações, debêntures, CRIs, FIIs, etc.
# Precisamos filtrar só os FIIs.
#
# Como funciona o filtro?
# → df[condicao] retorna só as linhas onde a condição é True
# → str.contains() faz busca parcial (não precisa ser match exato)
# → na=False trata valores nulos como False (não quebra o filtro)

def filtrar_fiis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra o DataFrame para manter apenas ofertas de FII.

    Também faz limpeza básica:
    - Remove colunas completamente vazias
    - Converte colunas de data para datetime
    - Ordena por data de registro (mais recente primeiro)
    """
    print(f"\nColunas disponíveis no CSV:")
    print(f"  {list(df.columns)}")

    # ── Detecta coluna oficial da CVM ──
    col_categoria = None

    for candidato in ["Tipo_Ativo", "TP_ATIVO", "TIPO_ATIVO"]:
        if candidato in df.columns:
            col_categoria = candidato
            print(f"  → Usando coluna oficial: {candidato}")
            break

    if not col_categoria:
        print("⚠️ Nenhuma coluna de ativo encontrada.")
        return df

    if "Tipo_Ativo" in df.columns:

        # print(df["Tipo_Ativo"].value_counts().head(20))

        df_fii = df[
            df["Tipo_Ativo"]
            .astype(str)
            .str.contains("IMOBILI", case=False, na=False)
        ].copy()

    else:

        print("⚠️ Tipo_Ativo não encontrada. Fallback por Nome_Emissor.")

        df_fii = df[
            df["Nome_Emissor"]
            .astype(str)
            .str.contains("FII", case=False, na=False)
        ].copy()

    
    print(f"\n  Total de ofertas no CSV: {len(df):,}")
    print(f"  Ofertas de FII:          {len(df_fii):,}")

    # ── Limpeza ──
    # Remove colunas 100% vazias (não agregam nada)
    df_fii = df_fii.dropna(axis=1, how="all")

    # Converte colunas de data (tentativa — o formato pode variar)
    for col in df_fii.columns:
        if "DT_" in col or "DATA" in col.upper():
            try:
                df_fii[col] = pd.to_datetime(df_fii[col], errors="coerce")
            except Exception:
                pass

    # Ordena por data de registro (mais recente primeiro)
    for col_data in ["DT_REG", "DT_REGISTRO", "DATA_REGISTRO"]:
        if col_data in df_fii.columns:
            df_fii = df_fii.sort_values(col_data, ascending=False)
            break

    return df_fii


# ─── BLOCO 4: Resumo analítico ────────────────────────────────────────────────
#
# Antes de salvar, imprimimos um resumo do que foi coletado.
# Isso é útil para o agente ter contexto histórico e para você
# validar que os dados fazem sentido.

def imprimir_resumo(df: pd.DataFrame):
    """Imprime estatísticas básicas sobre os FIIs coletados."""
    print(f"\n{'='*60}")
    print(f"  RESUMO — FIIs no histórico da CVM")
    print(f"{'='*60}")
    print(f"  Total de registros: {len(df):,}")

    # Top coordenadores/líderes (se a coluna existir)
    for col in ["NM_LIDER", "LIDER", "COORDENADOR_LIDER"]:
        if col in df.columns:
            print(f"\n  Top 10 coordenadores líderes:")
            # print(df[col].value_counts().head(10).to_string())
            break

    # Distribuição por ano
    for col in ["DT_REG", "DT_REGISTRO"]:
        if col in df.columns:
            anos = pd.to_datetime(df[col], errors="coerce").dt.year
            anos_validos = anos.dropna().astype(int)
            if not anos_validos.empty:
                print(f"\n  Registros por ano (últimos 10):")
                # print(anos_validos.value_counts().sort_index(ascending=False).head(10).to_string())
            break

    # Volume total
    for col in ["VL_TOTAL", "VALOR_TOTAL", "VL_OFERTA"]:
        if col in df.columns:
            total = pd.to_numeric(df[col], errors="coerce").sum()
            print(f"\n  Volume total histórico: R$ {total:,.0f}")
            break


# ─── BLOCO 5: Salvar ─────────────────────────────────────────────────────────

def salvar(df: pd.DataFrame, prefixo: str = "fiis_historico") -> Path:
    """Salva o DataFrame filtrado em CSV dentro de data/cvm/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = DATA_DIR / f"{prefixo}_{ts}.csv"

    # utf-8-sig = UTF-8 com BOM, que o Excel abre corretamente
    df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")

    print(f"\nCSV salvo: {csv_path}")
    print(f"Shape: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    return csv_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baixa histórico de FIIs do portal de dados abertos da CVM"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mostra resumo sem salvar o arquivo",
    )
    parser.add_argument(
        "--ano",
        type=int,
        default=None,
        help="Filtrar por ano específico (ex: --ano 2024)",
    )
    args = parser.parse_args()

    # 1. Baixa o ZIP
    buffer = baixar_zip()

    # 2. Extrai e lê o CSV
    df = extrair_csv(buffer)

    # 3. Filtra só FIIs
    df_fii = filtrar_fiis(df)

    # 4. Filtro adicional por ano (opcional)
    if args.ano:
        for col in ["DT_REG", "DT_REGISTRO", "DATA_REGISTRO"]:
            if col in df_fii.columns:
                anos = pd.to_datetime(df_fii[col], errors="coerce").dt.year
                df_fii = df_fii[anos == args.ano]
                print(f"\n  Filtrado para {args.ano}: {len(df_fii):,} registros")
                break

    # 5. Resumo
    imprimir_resumo(df_fii)

    # 6. Salva (a menos que seja só preview)
    if not args.preview:
        salvar(df_fii)
    else:
        print("\n[preview] Arquivo não salvo (use sem --preview para salvar)")


if __name__ == "__main__":
    main()