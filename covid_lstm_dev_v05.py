"""
Created on Mon Mar  8 15:11:51 2021

Baseado nos treinamentos realizados por:

https://github.com/TannerGilbert/Tutorials/blob/master/Keras-Tutorials/5.%20Stock%20Price%20Prediction%20using%20a%20Recurrent%20Neural%20Network/Stock%20Price%20Prediction.ipynb
https://github.com/randerson112358/Python/blob/master/LSTM_Stock/LSTM2.ipynb
https://towardsdatascience.com/line-chart-animation-with-plotly-on-jupyter-e19c738dc882

v01 - treinamento com o numero totais (acunmulativos de casos)
v02 - treinamento com o numero de casos diarios
v03 - treinamento com o numero totais e tratamento de arquivo
v04 - saida multi-steps - predidic varios resultados (multi steps)
v04 - alterar o grafico do v04
v05 - alteracao do arquivo base - utiliza treinamento v04

@author: lmmastella
"""

# %% Import libraries


import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

# %% create_dataset train


def create_df_train(data, steps_in=1, steps_out=1):
    """

    Parameters
    ----------
    data : DataFrame
        shape(n, 1)

    steps_in : int
        timesteps de amostras df para x

    steps_out : int
        numero de previsoes df para x, y


    Returns
    -------
    array x - predict
    array y - features

    """

    x, y = [], []
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + steps_in
        out_end_ix = end_ix + steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(data):
            break
        x.append(data[i:end_ix])
        y.append(data[end_ix:out_end_ix])
    return np.array(x), np.array(y)


# %% create_dataset predxict


def create_df_predict(data, steps_in):
    """

    Parameters
    ----------
    data : DataFrame
        shape(n, 1)

    steps_in : int
        timesteps de amostras df para x

    Returns
    -------
    array x - predict

    """

    x = []
    for i in range(steps_in, len(data)):
        x.append(data[i - steps_in: i, 0])
    return np.array(x)


# %% define_dataset


def define_df(tipo):
    """

    Parameters
    ----------
    tipo: str
        Casos ou Obitos

    Returns
    -------
    df (db)  cor1 e cor2 para os gráficos

    """

    if tipo == "Obitos":

        df = df_brasil[["obitosAcumulado"]]
        cor1 = "blue"
        cor2 = "red"

    else:

        df = df_brasil[["casosAcumulado"]]
        cor1 = "blue"
        cor2 = "royalblue"
    df.columns = [tipo]
    return (df, cor1, cor2)


# %% arquivo original baixado de https://covid.saude.gov.br


df1 = pd.read_csv("/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR1.csv", sep=";")
df2 = pd.read_csv("/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR2.csv", sep=";")

df_brasil_ori = pd.concat([df1, df2])

# df_brasil_ori = pd.read_csv(
#     "/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR.csv", sep=";"
# )
# limpar os campos sem dados
df_brasil_ori = df_brasil_ori.replace(np.nan, "", regex=True)
# arquivo original : df_brasil_ori


# %% Variáveis para tratar arquivos
# tipo - Casos ou Obitos
# tipo_local - regiao, estado ou municipio
# local - Brasil, RS, Porto Alegre etc.


tipo_local = "municipio"  # Pais/regiao, Estado/estado, Municipio/municipio
local = "Porto Alegre"
tipo = "Casos"  # 'Casos' ou 'Obitos'
day = datetime.today().strftime("%Y-%m-%d")  # dia do relatório (str)


# %% Preparar Dataset para seleção conforme as variáveis acima

# seleção
if tipo_local == "municipio":
    df_brasil = df_brasil_ori[df_brasil_ori[tipo_local] != ""]

else:
    df_brasil = df_brasil_ori[
        (df_brasil_ori[tipo_local] != "") & (df_brasil_ori["codmun"] == "")
    ]
    # problema dataset

# arquivo selecionado : df_brasil
# limpeza do arquivo com eliminação de colunas desnecessárias

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

# seleção
df_brasil = df_brasil[df_brasil[tipo_local] == local]
df_brasil = df_brasil.drop_duplicates(["data"]).set_index("data")

# %% Consertar dataset erradi

mask = (df_brasil['casosNovos'] > 120000) | (df_brasil['casosNovos'] < -120000)
df_brasil = df_brasil.loc[~mask]


# %%  Gráfico de casos reais do local selecionado

fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_brasil.index,
        y=df_brasil.casosAcumulado,
        mode="lines+markers",
        name="Casos Total",
        line_color="blue",
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=df_brasil.index,
        y=df_brasil.casosNovos,
        name="Casos Diarios",
        marker_color="blue",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=df_brasil.index,
        y=round(df_brasil.casosNovos.rolling(7).mean()),
        name=" MM7",
        marker_color="black",
    ),
    secondary_y=False,
)

# Criando Layout
fig.update_layout(
    title_text=local + " - Evolução de Casos",
    legend=dict(x=0.02, y=0.95),
    legend_orientation="v",
)

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos  Diarios", secondary_y=False)

graph = local + " LSTM - Gráfico Casos " + day + ".html"
py.plot(fig, filename=graph)

df_brasil.describe()
# %%  Gráfico de óbitos reais do local selecionado


fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_brasil.index,
        y=df_brasil.obitosAcumulado,
        mode="lines+markers",
        name="Óbitos Total",
        line_color="red",
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=df_brasil.index,
        y=df_brasil.obitosNovos,
        name="Óbitos Diarios",
        marker_color="red",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=df_brasil.index,
        y=round(df_brasil.obitosNovos.rolling(7).mean()),
        name=" MM7",
        marker_color="black",
    ),
    secondary_y=False,
)

# Criando Layout
fig.update_layout(
    title_text=local + " - Evolução de Óbitos",
    legend=dict(x=0.02, y=0.95),
    legend_orientation="v",
)

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Óbitos Total", secondary_y=True)
fig.update_yaxes(title_text="Óbitos  Diarios", secondary_y=False)

graph = local + " LSTM - Gráfico  Óbitos " + day + ".html"
py.plot(fig, filename=graph)


# %% Variáveis para database de treinamento


n_steps_in = 21  # amostras para LSTM
n_steps_out = 10  # Dias de previsao futura
n_features = 1  # previsao timeseries


# %% funcao que define o dataset dependendo do tipo (Casos ou Obitos)


df, cor1, cor2 = define_df(tipo)


# %% Preparar a base de dados de treinamento entre 0 e 1


df_scaler = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaler = scaler.fit_transform(df_scaler)


# %% Gerar o dataset para treinamento com a função create_dataset

data_x, data_y = create_df_train(df_scaler, n_steps_in, n_steps_out)


# %% Criar o modelo de deep learning LSTM

epochs = 500
batch = 16

model = Sequential()
model.add(LSTM(units=256, return_sequences=False,
          input_shape=(data_x.shape[1], 1)))
model.add(Dense(units=256))
model.add(Dense(units=n_steps_out))


# %% Compilar modelo

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])


# %% Treinar modelo se ainda não foi treinado(não tem o arquivo com o nome)

arq = "Covid_lstm_" + tipo + "_" + local + "_v_04.h5"

mcp = ModelCheckpoint(filepath=arq, monitor="val_loss",
                      save_best_only=True, verbose=1)

if not os.path.exists(arq):
    history = model.fit(
        data_x,
        data_y,
        epochs=epochs,
        batch_size=batch,
        validation_split=0.2,
        callbacks=[mcp],
    )
    model.save(arq)

model = load_model(arq)


# %% Verficacoes com o x_test - y_test os valores  loss e accuracy
#    Somente quando realizar o treinamento


loss, metric = model.evaluate(data_x, data_y)


# %% Graficos de performance
#    Somente quando realizar o treinamento


plt.figure(figsize=(8, 4))
plt.plot(history.history["mae"])
plt.plot(history.history["val_mae"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


# %% Predict actual - o mesmo do treinamento


data_pre = create_df_predict(df_scaler, n_steps_in)
data_pre = np.reshape(data_pre, (data_pre.shape[0], n_steps_in, n_features))

predictions_pre = model.predict(data_pre)
predictions_pre = scaler.inverse_transform(predictions_pre)
predictions_pre = np.around(predictions_pre).astype(int)
# predicoes dos valores atuais do df
predictions = predictions_pre[:, 0].reshape(-1, 1)
# predicoes dos valores futuros (n_steps_out)
predictions_p = predictions_pre[-1:, ].reshape(-1, 1)


# %% Acerto das datas (df_index) object data  tipo  2020-02-22  aaaa-mm-dd

# dataset original
index_days = [datetime.strptime(d, "%Y-%m-%d") for d in df.index[n_steps_in:]]

# predictions (pred_days
next_days = [index_days[-1] + timedelta(days=i)
             for i in range(1, n_steps_out + 1)]

# total
total_days = index_days + next_days


# %% Banco de dados de predicao


CasosReais = pd.Series(df[n_steps_in:].values[:, 0].astype(int))
CasosPre = pd.Series(np.concatenate(
    (predictions, predictions_p))[:, 0])  # Total
CasosDiasReais = pd.Series(CasosReais).diff()
CasosDiasPre = pd.Series(CasosPre).diff()
CasosMM7Reais = round(CasosDiasReais.rolling(7).mean(), 2)
CasosMM7Pre = round(CasosDiasPre.rolling(7).mean(), 2)

predict = (
    pd.DataFrame(
        [
            total_days,
            CasosReais,
            CasosPre,
            CasosDiasReais,
            CasosDiasPre,
            CasosMM7Reais,
            CasosMM7Pre,
        ],
        [
            "Data",
            "CasosReais",
            "CasosPre",
            "CasosDiasReais",
            "CasosDiasPre",
            "CasosMM7Reais",
            "CasosMM7Pre",
        ],
    )
    .transpose()
    .set_index("Data")
)


# %%  Gráfico de casos ou mortes reais e previstos do local selecionado

fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict.CasosReais,
        mode="lines+markers",
        name=tipo + " Reais",
        line_color=cor1,
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict.CasosPre,
        mode="lines+markers",
        name=tipo + " Previsto",
        line_color="crimson",
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=predict.index,
        y=predict.CasosDiasReais,
        name="Diario Reais",
        marker_color=cor2,
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Bar(
        x=predict.index,
        y=predict.CasosDiasPre,
        name="Diario Previsto",
        marker_color="tan",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict.CasosMM7Reais,
        name=tipo + " MM7 Reais",
        marker_color="black",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict.CasosMM7Pre,
        name=tipo + " MM7 Previsto",
        marker_color="red",
    ),
    secondary_y=False,
)

# Criando Layout
fig.update_layout(
    title_text=local + " Previsão e Evolução de " + tipo,
    legend=dict(x=0.02, y=0.95),
    legend_orientation="v",
)

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text=tipo + " Total", secondary_y=True)
fig.update_yaxes(title_text=tipo + " Diarios", secondary_y=False)

graph = local + " LSTM - Gráfico Previsão de " + tipo + " " + day + ".html"
py.plot(fig, filename=graph)

########################################################################
# NOVAS ANALISES
########################################################################

# %% arquivo original baixado de https://covid.saude.gov.br


df_brasil_ori = pd.read_csv(
    "/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR.csv", sep=";"
)
# limpar os campos sem dados
df_brasil_ori = df_brasil_ori.replace(np.nan, "", regex=True)
# arquivo original : df_brasil_ori

# %% variaveis

tipo_local = "municipio"  # Pais/regiao, Estado/estado, Municipio/municipio
tipo = "casosAcumulado"  # 'Casos' ou 'Obitos'

# %% preparando dataset

if tipo_local == "municipio":
    df_brasil = df_brasil_ori[df_brasil_ori[tipo_local] != ""]
if tipo_local == "estado":
    df_brasil = df_brasil_ori[(df_brasil_ori[tipo_local] != "")
                              & (df_brasil_ori['municipio'] == "")]

# %%

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

# %% seleção

df_brasil = df_brasil.reset_index(drop=True)
# df_brasil = df_brasil.drop_duplicates(["data"]).set_index("data")

# %% Analise comparativa estados

fig = px.line(df_brasil, x="data", y="casosAcumulado", color=tipo_local)
py.plot(fig)
# Estados repetidos

# %% colocando a visualizacao de logaritma e linear

fig = px.line(df_brasil, x="data", y="casosAcumulado", color=tipo_local)

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
# Estados repetidos


# %% Preparar o ambiente para animacao de barra e scatter

day = datetime.today().strftime("%Y-%m-%d")
includes = ["Porto Alegre", "Florianópolis", "Curitiba", "São Paulo"]
data_ini = "2021-01-01"
data_fin = day
tipo = "casosNovos"  # 'casosAcumulado', 'casosNovos','obitosAcumulado', 'obitosNovos'


# %%

df_brasil["data"] = pd.to_datetime(df_brasil["data"]).dt.strftime("%Y-%m-%d")
df_brasil = df_brasil[df_brasil[tipo].isin(includes)]
df_brasil = df_brasil[
    (df_brasil["data"] > data_ini) & (df_brasil["data"] < data_fin)
]
# %% Grafico animado de barra


fig = px.bar(
    df_brasil,
    x=tipo_local,
    y=tipo,
    color=tipo_local,
    animation_frame="data",
    animation_group=tipo_local,
    range_y=[0, 30000],
    title="Numero de casos por dia",
)
py.plot(fig)

# %% Grafico animado de bolas


df_brasil = [df_brasil[df_brasil] > 0]  # negativo invalido
fig = px.scatter(
    df_brasil,
    x="data",
    y=tipo,
    color=tipo,
    animation_frame="data",
    animation_group=tipo_local,
    size=tipo,
    range_y=[0, 30000],
    range_x=["2020-12-20", "2021-03-30"],
    title="Numero de casos por dia",
)
py.plot(fig)


# %%  Analise comparativa

# includes_ani=['RS', 'SC', 'PR', 'SP']
RS = df_brasil[df_brasil[tipo_local].isin(["RS"])]
SC = df_brasil[df_brasil[tipo_local].isin(["SC"])]
PR = df_brasil[df_brasil[tipo_local].isin(["PR"])]
SP = df_brasil[df_brasil[tipo_local].isin(["SP"])]


# %% Grafico
trace1 = go.Scatter(
    x=RS["data"][:2], y=RS[tipo][:2], mode="lines", line=dict(width=1.5), name="RS"
)
trace2 = go.Scatter(
    x=SC["data"][:2], y=SC[tipo][:2], mode="lines", line=dict(width=1.5), name="SC"
)
trace3 = go.Scatter(
    x=PR["data"][:2], y=PR[tipo][:2], mode="lines", line=dict(width=1.5), name="PR"
)
trace4 = go.Scatter(
    x=SP["data"][:2], y=SP[tipo][:2], mode="lines", line=dict(width=1.5), name="SP"
)

frames = [
    dict(
        data=[
            dict(type="scatter", x=RS["data"][: k + 1], y=RS[tipo][: k + 1]),
            dict(type="scatter", x=SC["data"][: k + 1], y=SC[tipo][: k + 1]),
            dict(type="scatter", x=PR["data"][: k + 1], y=PR[tipo][: k + 1]),
            dict(type="scatter", x=SP["data"][: k + 1], y=SP[tipo][: k + 1]),
        ],
        traces=[0, 1, 2, 3],
    )
    for k in range(1, len(SC) - 1)
]

# %%
layout = go.Layout(
    width=700,
    height=600,
    showlegend=False,
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1.05,
            x=1.15,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=3, redraw=False),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                )
            ],
        ),
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
layout.update(
    xaxis=dict(range=[data_ini, data_fin], autorange=False),
    yaxis=dict(range=[0, 35000], autorange=False),
)
fig = go.Figure(data=[trace1, trace2, trace3, trace4],
                frames=frames, layout=layout)
py.plot(fig)
# py.plot(fig, filename="Comparar.html")


# %%  Analise comparativa original

df = pd.read_csv(
    "https://raw.githubusercontent.com/shinokada/covid-19-stats/master/data/daily-new-confirmed-cases-of-covid-19-tests-per-case.csv"
)
df.columns = ["Country", "Code", "Date", "Confirmed", "Days since confirmed"]
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = df[(df["Date"] > "2020-03-15") & (df["Date"] < "2020-06-14")]
usa = df[df["Country"].isin(["United States"])]
brazil = df[df["Country"].isin(["Brazil"])]
india = df[df["Country"].isin(["India"])]
russia = df[df["Country"].isin(["Russia"])]
trace1 = go.Scatter(
    x=usa["Date"][:2], y=usa["Confirmed"][:2], mode="lines", line=dict(width=1.5)
)
trace2 = go.Scatter(
    x=brazil["Date"][:2], y=brazil["Confirmed"][:2], mode="lines", line=dict(width=1.5)
)
trace3 = go.Scatter(
    x=india["Date"][:2], y=india["Confirmed"][:2], mode="lines", line=dict(width=1.5)
)
trace4 = go.Scatter(
    x=russia["Date"][:2], y=russia["Confirmed"][:2], mode="lines", line=dict(width=1.5)
)
frames = [
    dict(
        data=[
            dict(type="scatter", x=usa["Date"]
                 [: k + 1], y=usa["Confirmed"][: k + 1]),
            dict(
                type="scatter",
                x=brazil["Date"][: k + 1],
                y=brazil["Confirmed"][: k + 1],
            ),
            dict(
                type="scatter", x=india["Date"][: k + 1], y=india["Confirmed"][: k + 1]
            ),
            dict(
                type="scatter",
                x=russia["Date"][: k + 1],
                y=russia["Confirmed"][: k + 1],
            ),
        ],
        traces=[0, 1, 2, 3],
    )
    for k in range(1, len(usa) - 1)
]
layout = go.Layout(
    width=700,
    height=600,
    showlegend=False,
    hovermode="x unified",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1.05,
            x=1.15,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=3, redraw=False),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                )
            ],
        ),
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
layout.update(
    xaxis=dict(range=["2020-03-16", "2020-06-13"], autorange=False),
    yaxis=dict(range=[0, 35000], autorange=False),
)
fig = go.Figure(data=[trace1, trace2, trace3, trace4],
                frames=frames, layout=layout)
# fig.show()
py.plot(fig, filename="Comparar.html")

# %%
