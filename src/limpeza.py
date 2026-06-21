# ============================================================
# limpeza.py
# Responsável por carregar, limpar e enriquecer o dataset
# de gastos pessoais antes de qualquer análise ou visualização.
# ============================================================

import pandas as pd
import numpy as np


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Lê o arquivo CSV e retorna um DataFrame bruto (ainda sujo).
    Recebe o caminho do arquivo como parâmetro.
    """
    df = pd.read_csv(caminho)
    return df


def limpar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica todas as etapas de limpeza no DataFrame e devolve:
      - df_limpo  : o DataFrame pronto para análise
      - relatorio : dicionário com os números de cada limpeza feita
                    (usado para mostrar na interface o que foi removido)
    """

    # Guardamos o tamanho original para calcular o que foi removido
    total_original = len(df)

    # ── 1. DUPLICATAS ────────────────────────────────────────────
    # Linhas completamente iguais não agregam informação nenhuma.
    qtd_duplicatas = df.duplicated().sum()
    df = df.drop_duplicates()

    # ── 2. CONVERSÃO DA COLUNA 'valor' PARA NUMÉRICO ─────────────
    # pd.to_numeric com errors='coerce' transforma qualquer coisa
    # que não seja número em NaN — por exemplo, a célula "abc" do CSV.
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')

    # ── 3. REMOÇÃO DE LINHAS COM VALOR NULO OU INVÁLIDO ──────────
    # Inclui tanto valores que já eram nulos quanto os que viraram
    # NaN após a conversão acima (como a linha "INVALIDO").
    qtd_nulos = df['valor'].isna().sum()
    df = df.dropna(subset=['valor'])

    # ── 4. FILTRA CATEGORIAS INVÁLIDAS ───────────────────────────
    # Categorias fora de um conjunto esperado são descartadas.
    categorias_validas = [
        'Alimentação', 'Transporte', 'Moradia', 'Saúde',
        'Lazer', 'Vestuário', 'Educação'
    ]
    qtd_categoria_invalida = (~df['categoria'].isin(categorias_validas)).sum()
    df = df[df['categoria'].isin(categorias_validas)]

    # ── 5. CONVERSÃO DA COLUNA 'data' PARA DATETIME ───────────────
    # Precisamos do tipo correto para filtros por mês/ano depois.
    df['data'] = pd.to_datetime(df['data'])

    # ── 6. CONVERSÃO DA COLUNA 'parcelas' PARA INTEIRO ────────────
    df['parcelas'] = pd.to_numeric(df['parcelas'], errors='coerce').fillna(1).astype(int)

    # ── 7. ENGENHARIA DE FEATURES (novas colunas úteis) ───────────
    # Extraímos mês e ano para facilitar agrupamentos temporais.
    df['mes']        = df['data'].dt.month
    df['ano']        = df['data'].dt.year
    df['mes_ano']    = df['data'].dt.to_period('M').astype(str)  # ex.: "2024-01"
    df['nome_mes']   = df['data'].dt.strftime('%b/%Y')           # ex.: "Jan/2024"

    # Criamos uma flag para identificar gastos parcelados
    df['parcelado'] = np.where(df['parcelas'] > 1, 'Sim', 'Não')

    # Calculamos o valor mensal equivalente de cada compra parcelada
    df['valor_mensal'] = df['valor'] / df['parcelas']

    # ── 8. RESET DO ÍNDICE ────────────────────────────────────────
    df = df.reset_index(drop=True)

    # ── 9. RELATÓRIO DE LIMPEZA ───────────────────────────────────
    total_removido = total_original - len(df)
    relatorio = {
        'total_original'       : total_original,
        'duplicatas_removidas' : int(qtd_duplicatas),
        'nulos_removidos'      : int(qtd_nulos),
        'invalidos_removidos'  : int(qtd_categoria_invalida),
        'total_removido'       : int(total_removido),
        'total_final'          : len(df),
    }

    return df, relatorio


def preparar_dados(caminho: str) -> tuple[pd.DataFrame, dict]:
    """
    Função principal do módulo: junta carregamento + limpeza.
    É ela que o app.py vai chamar.
    """
    df_bruto  = carregar_dados(caminho)
    df_limpo, relatorio = limpar_dados(df_bruto)
    return df_limpo, relatorio