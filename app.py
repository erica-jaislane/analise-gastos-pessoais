# ============================================================
# app.py  —  Interface principal do projeto
# Para rodar: streamlit run app.py
#
# Estrutura de abas:
#   Aba 1 → Relatório de Limpeza  (Critério 1 da rubrica)
#   Aba 2 → Visão Geral           (Critério 2)
#   Aba 3 → Análise por Categoria (Critério 2)
#   Aba 4 → Evolução Temporal     (Critério 2)
#   Aba 5 → Modelo Preditivo      (Critério opcional)
# ============================================================

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # sem janela gráfica, necessário para Streamlit

# Garante que a pasta 'src' seja encontrada independente de onde
# o script é executado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from limpeza import preparar_dados
from modelo  import treinar_modelo, grafico_previsao

# ── CONFIGURAÇÃO DA PÁGINA ────────────────────────────────────
st.set_page_config(
    page_title='Análise de Gastos Pessoais',
    page_icon='💰',
    layout='wide'
)

# ── TÍTULO PRINCIPAL ──────────────────────────────────────────
st.title('💰 Análise de Gastos Pessoais')
st.markdown('Explore seus hábitos financeiros e descubra onde o dinheiro vai.')
st.divider()

# ── CARREGAMENTO DOS DADOS ────────────────────────────────────
# Construímos o caminho do CSV de forma dinâmica para funcionar
# em qualquer máquina, independente de onde a pasta esteja salva.
CAMINHO_CSV = os.path.join(os.path.dirname(__file__), 'dados', 'gastos.csv')

# Usamos cache do Streamlit para não recarregar o CSV toda vez
# que o usuário interage com um filtro.
@st.cache_data
def carregar():
    return preparar_dados(CAMINHO_CSV)

df, relatorio = carregar()

# ── FILTROS LATERAIS ──────────────────────────────────────────
st.sidebar.header('🔍 Filtros')

# Filtro de Mês/Ano
meses_disponiveis = sorted(df['mes_ano'].unique())
meses_selecionados = st.sidebar.multiselect(
    'Selecione os meses:',
    options=meses_disponiveis,
    default=meses_disponiveis   # todos selecionados por padrão
)

# Filtro de Categoria
categorias_disponiveis = sorted(df['categoria'].unique())
categorias_selecionadas = st.sidebar.multiselect(
    'Selecione as categorias:',
    options=categorias_disponiveis,
    default=categorias_disponiveis
)

# Filtro de Forma de Pagamento
formas_disponiveis = sorted(df['forma_pagamento'].unique())
formas_selecionadas = st.sidebar.multiselect(
    'Forma de pagamento:',
    options=formas_disponiveis,
    default=formas_disponiveis
)

# Aplicamos todos os filtros de uma vez
df_filtrado = df[
    (df['mes_ano'].isin(meses_selecionados)) &
    (df['categoria'].isin(categorias_selecionadas)) &
    (df['forma_pagamento'].isin(formas_selecionadas))
]

# Avisa se os filtros zeraram os dados
if df_filtrado.empty:
    st.warning('⚠️ Nenhum dado encontrado com os filtros selecionados.')
    st.stop()

# ── ABAS PRINCIPAIS ───────────────────────────────────────────
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    '🧹 Limpeza dos Dados',
    '📊 Visão Geral',
    '🗂️ Por Categoria',
    '📅 Evolução Temporal',
    '🤖 Modelo Preditivo',
])


# ════════════════════════════════════════════════════════════════
# ABA 1 — RELATÓRIO DE LIMPEZA
# Mostra o que foi feito no pré-processamento, transparente e claro
# ════════════════════════════════════════════════════════════════
with aba1:
    st.subheader('🧹 Etapas de Limpeza e Preparação dos Dados')
    st.markdown(
        'Antes de qualquer análise, os dados passaram por um processo de '
        'limpeza para garantir qualidade e confiabilidade nos resultados.'
    )

    # Métricas do relatório
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Registros originais',    relatorio['total_original'])
    col2.metric('Duplicatas removidas',   relatorio['duplicatas_removidas'])
    col3.metric('Valores nulos/inválidos',relatorio['nulos_removidos'] + relatorio['invalidos_removidos'])
    col4.metric('Registros após limpeza', relatorio['total_final'])

    st.divider()

    # Detalhamento das etapas
    st.markdown('### 📋 O que foi feito?')

    etapas = [
        ('1️⃣ Remoção de duplicatas',
         f"Encontradas e removidas **{relatorio['duplicatas_removidas']} linha(s) duplicada(s)** "
         f"(registros idênticos que apareciam mais de uma vez)."),

        ('2️⃣ Conversão do tipo da coluna `valor`',
         'A coluna `valor` foi convertida de texto para número real (`float`). '
         'Valores que não puderam ser convertidos (ex.: "abc") viraram `NaN`.'),

        ('3️⃣ Remoção de valores nulos',
         f"**{relatorio['nulos_removidos']} registro(s)** com valor ausente ou ilegível foram descartados."),

        ('4️⃣ Filtragem de categorias inválidas',
         f"**{relatorio['invalidos_removidos']} registro(s)** com categoria fora do padrão (ex.: '???') foram removidos."),

        ('5️⃣ Conversão da coluna `data` para datetime',
         'Necessário para extrair mês, ano e ordenar cronologicamente.'),

        ('6️⃣ Engenharia de features',
         'Novas colunas criadas: `mes`, `ano`, `mes_ano`, `nome_mes`, `parcelado`, `valor_mensal`. '
         'Elas facilitam agrupamentos e enriquecem a análise.'),
    ]

    for titulo, descricao in etapas:
        with st.expander(titulo):
            st.markdown(descricao)

    st.divider()
    st.markdown('### 👀 Amostra dos dados limpos')
    st.dataframe(df.head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# ABA 2 — VISÃO GERAL
# KPIs + gráfico de pizza das categorias
# ════════════════════════════════════════════════════════════════
with aba2:
    st.subheader('📊 Visão Geral dos Gastos')

    total_gasto    = df_filtrado['valor'].sum()
    media_por_mes  = df_filtrado.groupby('mes_ano')['valor'].sum().mean()
    maior_gasto    = df_filtrado.loc[df_filtrado['valor'].idxmax()]
    qtd_transacoes = len(df_filtrado)

    # KPIs no topo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('💸 Total gasto',        f'R$ {total_gasto:,.2f}')
    c2.metric('📆 Média mensal',       f'R$ {media_por_mes:,.2f}')
    c3.metric('🏷️ Transações',         qtd_transacoes)
    c4.metric('🔺 Maior gasto único',  f'R$ {maior_gasto["valor"]:,.2f}')

    st.markdown(f"> 🔺 Maior gasto: **{maior_gasto['descricao']}** ({maior_gasto['categoria']}) em {maior_gasto['data'].strftime('%d/%m/%Y')}")

    st.divider()

    # ── Gráfico de Pizza: participação de cada categoria ──────────
    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown('#### Distribuição por Categoria')
        por_categoria = df_filtrado.groupby('categoria')['valor'].sum()

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        cores = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1']
        wedges, texts, autotexts = ax1.pie(
            por_categoria,
            labels=por_categoria.index,
            autopct='%1.1f%%',
            colors=cores[:len(por_categoria)],
            startangle=140,
            pctdistance=0.82
        )
        # Deixamos os percentuais em negrito para melhor leitura
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax1.set_title('Participação de cada categoria no total', fontsize=12)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_dir:
        st.markdown('#### Total por Categoria (R$)')
        # Tabela resumida ao lado da pizza
        tabela = (
            por_categoria
            .reset_index()
            .rename(columns={'valor': 'Total (R$)'})
            .sort_values('Total (R$)', ascending=False)
        )
        tabela['Total (R$)'] = tabela['Total (R$)'].apply(lambda x: f'R$ {x:,.2f}')
        st.dataframe(tabela, use_container_width=True, hide_index=True)

    st.divider()

    # ── Gráfico de Barras: forma de pagamento ─────────────────────
    st.markdown('#### Gastos por Forma de Pagamento')
    por_pagamento = df_filtrado.groupby('forma_pagamento')['valor'].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    bars = ax2.bar(por_pagamento.index, por_pagamento.values, color='steelblue', edgecolor='white')

    # Rótulo de valor em cima de cada barra
    for bar in bars:
        altura = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            altura + 20,
            f'R$ {altura:,.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    ax2.set_title('Total gasto por forma de pagamento', fontsize=12)
    ax2.set_xlabel('Forma de Pagamento')
    ax2.set_ylabel('Total (R$)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ════════════════════════════════════════════════════════════════
# ABA 3 — ANÁLISE POR CATEGORIA
# Detalhes de cada categoria selecionável
# ════════════════════════════════════════════════════════════════
with aba3:
    st.subheader('🗂️ Análise por Categoria')

    categoria_escolhida = st.selectbox(
        'Escolha uma categoria para detalhar:',
        options=sorted(df_filtrado['categoria'].unique())
    )

    df_cat = df_filtrado[df_filtrado['categoria'] == categoria_escolhida]

    # Métricas da categoria
    c1, c2, c3 = st.columns(3)
    c1.metric('Total gasto',      f'R$ {df_cat["valor"].sum():,.2f}')
    c2.metric('Qtd transações',   len(df_cat))
    c3.metric('Ticket médio',     f'R$ {df_cat["valor"].mean():,.2f}')

    st.divider()

    col_a, col_b = st.columns(2)

    # ── Barras horizontais: maiores gastos individuais ─────────────
    with col_a:
        st.markdown(f'#### Maiores gastos em {categoria_escolhida}')
        top10 = df_cat.nlargest(10, 'valor')[['data','descricao','valor']].copy()
        top10['data'] = top10['data'].dt.strftime('%d/%m/%Y')
        top10['valor_fmt'] = top10['valor'].apply(lambda x: f'R$ {x:,.2f}')

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        cores_bar = plt.cm.Blues_r([i / len(top10) * 0.6 + 0.2 for i in range(len(top10))])
        ax3.barh(top10['descricao'], top10['valor'], color=cores_bar, edgecolor='white')
        ax3.set_xlabel('Valor (R$)')
        ax3.set_title(f'Top gastos — {categoria_escolhida}', fontsize=11)
        ax3.invert_yaxis()  # maior no topo
        ax3.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Evolução mensal da categoria ───────────────────────────────
    with col_b:
        st.markdown(f'#### Evolução mensal — {categoria_escolhida}')
        evolucao_cat = df_cat.groupby('mes_ano')['valor'].sum()

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(
            evolucao_cat.index, evolucao_cat.values,
            marker='s', color='darkorange', linewidth=2
        )
        ax4.fill_between(
            evolucao_cat.index, evolucao_cat.values,
            alpha=0.15, color='darkorange'
        )
        ax4.set_title(f'Gasto mensal — {categoria_escolhida}', fontsize=11)
        ax4.set_xlabel('Mês')
        ax4.set_ylabel('Total (R$)')
        ax4.tick_params(axis='x', rotation=30)
        ax4.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.divider()
    st.markdown('#### 📋 Todos os registros desta categoria')
    st.dataframe(
        df_cat[['data','descricao','valor','forma_pagamento','parcelas','parcelado']]
        .sort_values('data', ascending=False)
        .assign(data=lambda x: x['data'].dt.strftime('%d/%m/%Y'))
        .assign(valor=lambda x: x['valor'].apply(lambda v: f'R$ {v:,.2f}')),
        use_container_width=True,
        hide_index=True
    )


# ════════════════════════════════════════════════════════════════
# ABA 4 — EVOLUÇÃO TEMPORAL
# Gastos totais mês a mês e comparativo entre categorias
# ════════════════════════════════════════════════════════════════
with aba4:
    st.subheader('📅 Evolução Temporal dos Gastos')

    # ── Gráfico de linha: total mensal ────────────────────────────
    total_mensal = df_filtrado.groupby('mes_ano')['valor'].sum()

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(
        total_mensal.index, total_mensal.values,
        marker='o', color='steelblue', linewidth=2.5
    )
    ax5.fill_between(total_mensal.index, total_mensal.values, alpha=0.15, color='steelblue')

    # Anotação do valor em cada ponto
    for x, y in zip(total_mensal.index, total_mensal.values):
        ax5.annotate(
            f'R${y:,.0f}',
            (x, y),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center', fontsize=8, color='steelblue'
        )

    ax5.set_title('Total de Gastos por Mês', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Mês')
    ax5.set_ylabel('Total (R$)')
    ax5.tick_params(axis='x', rotation=30)
    ax5.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.divider()

    # ── Barras empilhadas: categorias por mês ─────────────────────
    st.markdown('#### Composição mensal por categoria')
    pivot = df_filtrado.pivot_table(
        index='mes_ano', columns='categoria', values='valor', aggfunc='sum', fill_value=0
    )

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    cores_cat = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1']
    pivot.plot(kind='bar', stacked=True, ax=ax6, color=cores_cat[:len(pivot.columns)], edgecolor='white')

    ax6.set_title('Gastos mensais por categoria (empilhado)', fontsize=12)
    ax6.set_xlabel('Mês')
    ax6.set_ylabel('Total (R$)')
    ax6.legend(loc='upper right', fontsize=8, title='Categoria')
    ax6.tick_params(axis='x', rotation=30)
    ax6.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    st.divider()

    # ── Heatmap manual: intensidade de gastos por mês/categoria ───
    st.markdown('#### 🗺️ Mapa de calor — Gastos por Categoria e Mês')

    fig7, ax7 = plt.subplots(figsize=(10, 4))
    im = ax7.imshow(pivot.T.values, cmap='YlOrRd', aspect='auto')

    ax7.set_xticks(range(len(pivot.index)))
    ax7.set_yticks(range(len(pivot.columns)))
    ax7.set_xticklabels(pivot.index, rotation=30, ha='right', fontsize=9)
    ax7.set_yticklabels(pivot.columns, fontsize=9)
    ax7.set_title('Intensidade de gastos (R$) por categoria e mês', fontsize=11)

    # Adicionamos os valores dentro de cada célula
    for i in range(len(pivot.columns)):
        for j in range(len(pivot.index)):
            val = pivot.T.values[i, j]
            ax7.text(j, i, f'{val:,.0f}', ha='center', va='center', fontsize=7,
                     color='black' if val < pivot.values.max() * 0.6 else 'white')

    plt.colorbar(im, ax=ax7, label='R$')
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close(fig7)


# ════════════════════════════════════════════════════════════════
# ABA 5 — MODELO PREDITIVO
# Regressão Linear para prever o próximo mês
# ════════════════════════════════════════════════════════════════
with aba5:
    st.subheader('🤖 Modelo Preditivo de Gastos')
    st.markdown(
        'Utilizamos **Regressão Linear** (Scikit-learn) para identificar a tendência '
        'de gastos ao longo dos meses e prever o total do próximo mês.'
    )

    resultado = treinar_modelo(df_filtrado)

    if 'erro' in resultado:
        st.warning(resultado['erro'])
    else:
        # Métricas do modelo
        c1, c2, c3 = st.columns(3)
        c1.metric('📈 Previsão próximo mês', f'R$ {resultado["previsao_proximo"]:,.2f}')
        c2.metric('📉 Erro médio (MAE)',      f'R$ {resultado["mae"]:,.2f}')
        c3.metric('📐 R² (qualidade)',        resultado['r2'])

        st.markdown(
            '> **MAE** (*Mean Absolute Error*): em média, o modelo erra R$ '
            f'{resultado["mae"]:,.2f} por mês.  \n'
            '> **R²**: quanto mais próximo de 1.0, melhor o modelo explica a variação dos gastos.'
        )

        st.divider()

        # Gráfico de previsão
        fig_modelo = grafico_previsao(resultado)
        st.pyplot(fig_modelo)
        plt.close(fig_modelo)

        st.divider()

        # Tabela com os dados usados no treino
        st.markdown('#### 📋 Série histórica mensal usada no modelo')
        serie_exibir = resultado['serie'][['mes_ano','total_gasto','mes_num']].copy()
        serie_exibir.columns = ['Mês', 'Total Gasto (R$)', 'Índice do mês']
        serie_exibir['Total Gasto (R$)'] = serie_exibir['Total Gasto (R$)'].apply(lambda x: f'R$ {x:,.2f}')
        st.dataframe(serie_exibir, use_container_width=True, hide_index=True)

        # Explicação didática do modelo
        with st.expander('📘 Como funciona a Regressão Linear aqui?'):
            st.markdown(
                '''
                1. **X (entrada):** número sequencial do mês (1 = primeiro mês do dataset, 2 = segundo, etc.)
                2. **y (saída):** total gasto naquele mês.
                3. O modelo encontra a **reta** que melhor passa pelos pontos históricos.
                4. Para prever o próximo mês, basta passar o próximo número sequencial.
                5. **Limitação:** o modelo assume que a tendência é linear — não captura sazonalidade ou eventos inesperados.
                '''
            )