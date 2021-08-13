########################################################################
# NOVAS ANALISES
########################################################################

# %% Import libraries


from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots


# %% arquivo original baixado de https://covid.saude.gov.br

df_brasil_ori = pd.read_csv(
    "/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR.csv", sep=";"
)
# limpar os campos sem dados
df_brasil_ori = df_brasil_ori.replace(np.nan, "", regex=True)
# arquivo original : df_brasil_ori

# %% variaveis para analise individual


tipo = 'Casos'            # Casos e Obitos
tipo_local = 'Estado'  # Pais/regiao, Estado, Municipio
local = 'RS'
local_mul = ['RS', 'SC', 'PR', 'MG']
day = datetime.today().strftime("%Y-%m-%d")
data_ini = '2021-01-01'
data_fin = day


# %% preparando dataset

if tipo_local == "Municipio":
    df_brasil = df_brasil_ori[df_brasil_ori['municipio'] != ""]
if tipo_local == "Estado":
    df_brasil = df_brasil_ori[(df_brasil_ori['estado'] != "")
                              & (df_brasil_ori['codmun'] == "")]

# %% prepara df

df_brasil = df_brasil.drop(
    columns=[
        "coduf",
        "codmun",
        "codRegiaoSaude",
        "nomeRegiaoSaude",
        "semanaEpi",
        "populacaoTCU2019",
        "Recuperadosnovos",
        "emAcompanhamentoNovos",
        "interior/metropolitana",
    ]
)

df_brasil.columns = ['Regiao', 'Estado', 'Municipio', 'data',
                     'Casos', 'casosNovos', 'Obitos', 'obitosNovos']

# %% seleção somente para analise individuais

df_ind = df_brasil[df_brasil[tipo_local] == local].set_index("data")


# %% grafico individual de Casos


fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_ind.index,
        y=df_ind[tipo],
        mode="lines+markers",
        name=tipo + " Total ",
        line_color="blue",
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=df_ind.index,
        y=df_ind.casosNovos,
        name=tipo + " Diarios ",
        marker_color="blue",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=df_ind.index,
        y=round(df_ind.casosNovos.rolling(7).mean()),
        name=" MM7",
        marker_color="black",
    ),
    secondary_y=False,
)

# Criando Layout
fig.update_layout(
    title_text=local + " - Evolução de " + tipo,
    legend=dict(x=0.02, y=0.95),
    legend_orientation="v",
)

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos  Diarios", secondary_y=False)

graph = local + " Gráfico de " + tipo + day + ".html"
py.plot(fig, filename=graph)

# %% grafico individual de Obitos


fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_ind.index,
        y=df_ind[tipo],
        mode="lines+markers",
        name=tipo + " Total ",
        line_color="red",
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=df_ind.index,
        y=df_ind.obitosNovos,
        name=tipo + " Diarios ",
        marker_color="red",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=df_ind.index,
        y=round(df_ind.obitosNovos.rolling(7).mean()),
        name=" MM7",
        marker_color="black",
    ),
    secondary_y=False,
)

# Criando Layout
fig.update_layout(
    title_text=local + " - Evolução de " + tipo,
    legend=dict(x=0.02, y=0.95),
    legend_orientation="v",
)

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos  Diarios", secondary_y=False)

graph = local + " Gráfico de Obitos " + day + ".html"
py.plot(fig, filename=graph)


# %% define df para analise multiplas

# df_mul = df_brasil[df_brasil.Estado.isin(local_mul)]
df_mul = df_brasil[df_brasil[tipo_local].isin(local_mul)].set_index("data")

# %% Analise comparativa de toods os estados

fig = px.line(df_brasil, x="data", y=tipo, color=tipo_local)
py.plot(fig)

# %% Analise comparativa estados selecionados

fig = px.line(df_mul, x=df_mul.index, y=tipo, color=tipo_local)
py.plot(fig)

# %% colocando a visualizacao de logaritma e linear

fig = px.line(df_mul, x=df_mul.index, y=tipo, color=tipo_local)

fig.update_layout(
    hovermode="x unified",  # mostra todos os dados no grafico
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list(
                [
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="LINEAR",
                        method="relayout",
                    ),
                    dict(args=[{"yaxis.type": "log"}],
                         label="LOG", method="relayout"),
                ]
            ),
        ),
    ],
)
py.plot(fig)


# %% Grafico animado de barra

tipo = 'casosNovos'
df_mul = df_mul[(df_mul.index > data_ini) & (df_mul.index < data_fin)]

fig = px.bar(
    df_mul,
    x=tipo_local,
    y=tipo,
    color=tipo_local,
    animation_frame=df_mul.index,
    animation_group=tipo_local,
    range_y=[0, 30000],
    title="Numero de casos por dia",
)
py.plot(fig)

# %% Grafico animado de bolas

tipo = 'casosNovos'

df_mul = df_mul[(df_mul.index > data_ini) & (df_mul.index < data_fin)]
df_mul = df_mul[df_mul[tipo] > 0]  # negativo invalido

fig = px.scatter(
    df_mul,
    x=df_mul.index,
    y=tipo,
    color=tipo_local,
    animation_frame=df_mul.index,
    animation_group=tipo_local,
    size=tipo,
    range_y=[0, 10000],
    range_x=[data_ini, data_fin],
    title="Numero de casos por dia",
)
py.plot(fig)
