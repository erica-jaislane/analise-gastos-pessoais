# ============================================================
# modelo.py
# Treina um modelo de Regressão Linear para prever o total
# de gastos do próximo mês com base no histórico mensal.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sem janela, compatível com Streamlit

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def preparar_serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa os dados por mês e calcula o total gasto em cada um.
    Também cria a coluna 'mes_num' (1, 2, 3...) que será o
    atributo (X) usado para treinar o modelo.
    """
    # Somamos o valor de todos os gastos de cada mês
    serie = (
        df.groupby('mes_ano')['valor']
        .sum()
        .reset_index()
        .rename(columns={'valor': 'total_gasto'})
    )

    # Ordenamos cronologicamente para o índice fazer sentido
    serie = serie.sort_values('mes_ano').reset_index(drop=True)

    # Criamos um número sequencial para o mês (1 = primeiro mês do dataset)
    serie['mes_num'] = np.arange(1, len(serie) + 1)

    return serie


def treinar_modelo(df: pd.DataFrame) -> dict:
    """
    Treina o modelo de Regressão Linear e devolve um dicionário
    com tudo que o app.py precisa: o modelo, as métricas e os dados.
    """
    serie = preparar_serie_temporal(df)

    # Precisamos de pelo menos 3 meses para treinar com sentido
    if len(serie) < 3:
        return {'erro': 'São necessários pelo menos 3 meses de dados.'}

    # X = número do mês (variável independente / feature)
    # y = total gasto naquele mês (variável que queremos prever)
    X = serie[['mes_num']]
    y = serie['total_gasto']

    # Dividimos em treino (80%) e teste (20%)
    # random_state garante que os resultados sejam reproduzíveis
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Instanciamos e treinamos o modelo
    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    # Fazemos as previsões no conjunto de teste
    y_pred = modelo.predict(X_teste)

    # ── MÉTRICAS DE AVALIAÇÃO ────────────────────────────────────
    mae = mean_absolute_error(y_teste, y_pred)
    r2  = r2_score(y_teste, y_pred) if len(y_teste) > 1 else None

    # ── PREVISÃO DO PRÓXIMO MÊS ──────────────────────────────────
    # O próximo mês será o número sequencial imediatamente após o último
    proximo_mes_num = np.array([[serie['mes_num'].max() + 1]])
    previsao_proximo = modelo.predict(proximo_mes_num)[0]

    return {
        'modelo'            : modelo,
        'serie'             : serie,
        'X_treino'          : X_treino,
        'X_teste'           : X_teste,
        'y_treino'          : y_treino,
        'y_teste'           : y_teste,
        'y_pred'            : y_pred,
        'mae'               : round(mae, 2),
        'r2'                : round(r2, 4) if r2 is not None else 'N/A',
        'previsao_proximo'  : round(previsao_proximo, 2),
    }


def grafico_previsao(resultado: dict) -> plt.Figure:
    """
    Gera o gráfico com a linha histórica + linha de previsão do modelo.
    Retorna um objeto Figure do Matplotlib para o Streamlit renderizar.
    """
    serie   = resultado['serie']
    modelo  = resultado['modelo']

    fig, ax = plt.subplots(figsize=(10, 4))

    # Linha dos valores reais históricos
    ax.plot(
        serie['mes_ano'],
        serie['total_gasto'],
        marker='o',
        color='steelblue',
        linewidth=2,
        label='Gasto real'
    )

    # Linha da tendência calculada pelo modelo (para todos os meses)
    y_tendencia = modelo.predict(serie[['mes_num']])
    ax.plot(
        serie['mes_ano'],
        y_tendencia,
        linestyle='--',
        color='tomato',
        linewidth=2,
        label='Tendência (modelo)'
    )

    # Ponto de previsão do próximo mês
    proximo_label = f"Próximo mês\nR$ {resultado['previsao_proximo']:,.2f}"
    ax.scatter(
        [serie['mes_ano'].iloc[-1]],   # posicionamos visualmente no último ponto
        [resultado['previsao_proximo']],
        color='gold',
        zorder=5,
        s=120,
        label=proximo_label
    )

    ax.set_title('Previsão de Gastos — Regressão Linear', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Total Gasto (R$)')
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig