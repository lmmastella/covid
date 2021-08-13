"""
Created on Mon Jan  6 15:11:51 2020

Baseado em:
https://github.com/TannerGilbert/Tutorials/blob/master/Keras-Tutorials/5.%20Stock%20Price%20Prediction%20using%20a%20Recurrent%20Neural%20Network/Stock%20Price%20Prediction.ipynb
https://github.com/randerson112358/Python/blob/master/LSTM_Stock/LSTM2.ipynb

v01 - treinamento com o numero totais (acunmulativos de casos)
v02 - treinamento com o numero de casos diarios

@author: lmmastella
"""

# %% Import libraries

import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta
import os


# %% Prepara os dados em time step para a LSTM(timesteps)

def create_dataset(data, ts=1):
    """

    Parameters
    ----------
    data : DataFrame
        shape(n, 1)

    ts : int
        timesteps

    Returns
    -------
    array x - predict
    array y - features

    """

    x, y = [], []
    for i in range(ts, len(data)):
        x.append(data[i-ts:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# %% Predict


def create_predict(data, pred_datas):
    """

    Parameters
    ----------
    data : array
        shape(n, 1)

    pred_datas : int
        number previsions

    Returns
    -------
    previsions (array) - predict features

    """

    previsions = []

    for i in range(pred_datas):
        x_input = data.reshape(1, n_steps, n_features)
        prevision = model.predict(x_input)
        previsions.append(prevision[0, 0])
        data = np.delete(data, 0, axis=0)
        data = np.vstack([data, prevision])
    return np.array(previsions)


# %% Define o tipo de análise - Casos ou Mortes

def define_dataset(tipo):
    """

    Parameters
    ----------
    tipo: str
        Casos ou Mortes


    Returns
    -------
    df (db)  cor1 e cor2 para os gráficos

    """

    if tipo == 'Mortes':

        df = df_mortes.iloc[:, 1:].T.sum(axis=1)
        cor1 = 'blue'
        cor2 = 'red'

    else:

        df = df_casos.iloc[:, 1:].T.sum(axis=1)
        cor1 = 'blue'
        cor2 = 'royalblue'
    return(df, cor1, cor2)

# %% Importa Dataset e prepara base de dados para análise


# arquivo original baixado de https://covid.saude.gov.br
df_brasil_ori = pd.read_csv('HIST_PAINEL_COVIDBR_19mar2021.csv', sep=';')
# limpar os campos sem dados
df_brasil_ori = df_brasil_ori.replace(np.nan, '', regex=True)
# arquivo original : df_brasil_ori


# %% Variáveis para tratar arquivos
# tipo - Casos ou Mortes
# tipo_local - regiao, estado ou municipio
# local - Brasil, RS, Porto Alegre etc.

tipo_local = 'regiao'  # Pais/regiao, Estado/estado, Municipio/municipio
local = 'Brasil'
tipo = 'Mortes'    # 'Casos' ou 'Mortes'
day = datetime.today().strftime("%Y-%m-%d")  # dia do relatório (str)

# %% Preparar Dataset para seleção conforme as variáveis acima

# seleção
if tipo_local == 'municipio':
    df_brasil = df_brasil_ori[df_brasil_ori[tipo_local] != '']

elif tipo_local == 'estado':
    df_brasil = df_brasil_ori[(df_brasil_ori[tipo_local] != '')
                              & (df_brasil_ori['codmun'] == '')] # problema dataset

elif tipo_local == 'regiao':
    df_brasil = df_brasil_ori[(df_brasil_ori[tipo_local] != '')
                              & (df_brasil_ori['codmun'] == '')] # problema dataset

# arquivo selecionado : df_brasil
# limpeza do arquivo com eliminação de colunas desnecessáriasbe datas duplicadas
df_brasil = df_brasil.drop(columns=['coduf', 'codmun',
                                    'codRegiaoSaude', 'nomeRegiaoSaude',
                                    'semanaEpi', 'populacaoTCU2019',
                                    'Recuperadosnovos', 'emAcompanhamentoNovos',
                                    'interior/metropolitana'])

# seleção
df_brasil = df_brasil[df_brasil[tipo_local] == local]

# preparar dataset para treinamento eliminando datas duplicada
df_brasil = df_brasil.drop_duplicates(['data'])

df_casos = df_brasil.pivot(index=tipo_local, columns='data',
                           values='casosAcumulado').reset_index().fillna(0)

df_mortes = df_brasil.pivot(index=tipo_local, columns='data',
                            values='obitosAcumulado').reset_index().fillna(0)


# %% Preparacao do dataset para training em LSTM e predicts

# acessa a funcao que define qual o daset dependendo do tipo (Casos ou Mortes)
df, cor1, cor2 = define_dataset(tipo)
# transforma series em dataframe com nome da coluna e inicio acima de 0
df = pd.DataFrame(df)
df.columns = [tipo]
# df = df.loc[df[tipo] > 0]


# %%  Prepara grafico de casos ou mortes reais do local selecionado

total = df.values[:, 0].astype(int)
dias = pd.Series(total).diff()
MM7 = round(dias.rolling(7).mean(), 2)
MM14 = round(dias.rolling(14).mean(), 2)

grafico = pd.DataFrame([df.index,
                        list(total),
                        list(dias),
                        list(MM7),
                        list(MM14)],
                       ["Data", "CasosTotal", "CasosDias",
                        "MM7", "MM14"]).\
                        transpose().set_index("Data")


# %%  grafico de casos ou mortes reais e previstos do local selecionado

fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
              go.Scatter(
                          x=grafico.index,
                          y=grafico['CasosTotal'],
                          mode='lines+markers',
                          name=tipo + ' Reais',
                          line_color=cor1),
              secondary_y=True
              )

fig.add_trace(
              go.Bar(x=grafico.index,
                     y=grafico['CasosDias'],
                     name='Diario Real',
                     marker_color=cor2),
              secondary_y=False
              )

fig.add_trace(
              go.Scatter(x=grafico.index,
                         y=grafico['MM7'],
                         name=tipo + ' MM7',
                         marker_color='black'),
              secondary_y=False
              )

fig.add_trace(
              go.Scatter(x=grafico.index,
                         y=grafico['MM14'],
                         name=tipo + ' MM14',
                         marker_color='green'),
              secondary_y=False
              )

# Criando Layout
fig.update_layout(title_text=local + ' - Evolução de ' + tipo,
                  legend=dict(x=0.02, y=0.95), legend_orientation="v")

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text=tipo + " Total", secondary_y=True)
fig.update_yaxes(title_text=tipo + " Diarios", secondary_y=False)

graph = local + ' LSTM - Grafico  ' + tipo + ' ' + day + '.html'
py.plot(fig, filename=graph)


# %% Variáveis para database de treinamento

n_steps = 10       # amostras para LSTM
n_features = 1     # target - y
pred_days = 10     # Dias de previsao futura
epochs = 2000
batch = 32


# %% Preparar a base de dados de treinamento entre 0 e 1

df_scaler = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaler = scaler.fit_transform(df_scaler)


# %% Gerar o dataset para treinamento com a função create_dataset

data_x, data_y = create_dataset(df_scaler, n_steps)


# %% Reshape features for LSTM Layer [samples, time steps, features]

data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))


# %% Criar o modelo de deep learning LSTM

model = Sequential()
model.add(LSTM(units=256, return_sequences=True,
               input_shape=(data_x.shape[1], 1)))
model.add(LSTM(units=256))
model.add(Dense(units=1))


# %% Compilar modelo

model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['mae'])


# %% Treinar modelo se ainda não foi treinado(não tem o arquivo com o nome)

arq = 'Covid_lstm_' + tipo + '_' + local + '.h5'

mcp = ModelCheckpoint(filepath=arq, monitor='loss',
                      save_best_only=True, verbose=1)

if(not os.path.exists(arq)):
    history = model.fit(data_x, data_y,
                        epochs=epochs,
                        batch_size=batch,
                        validation_split=0.1,
                        callbacks=[mcp])
    model.save(arq)

model = load_model(arq)
model.summary()


# %% Testar com o x_test - y_test os valores de precisão loss e mean_absolute_error


loss, metric = model.evaluate(data_x, data_y)


# %% Predict actual - o mesmo do treinamento

predictions = model.predict(data_x)
predictions = scaler.inverse_transform(predictions)
predictions = np.around(predictions).astype(int)

# %% Next days - ultimos n_steps do arquivo inicial shape (n_steps, 1)

df_p = df[-n_steps:].values    # ultimos valores (n_steps)
df_p = scaler.transform(df_p)  # prepara para o lstm

# faz a predição conformr o numero de dias (pred_days) e a função create_predict
predictions_p = create_predict(df_p, pred_days)  # previsoes
# retorna as valores nornais do dataset
predictions_p = scaler.inverse_transform(predictions_p.reshape(-1, 1))
predictions_p = np.around(predictions_p).astype(int)


# %% Acerto das datas (df_index) object data  tipo  2020-02-22  aaaa-mm-dd

# dataset original
index_days = [datetime.strptime(d, '%Y-%m-%d')
              for d in df.index[n_steps:]]

# predictions (pred_days
next_days = [index_days[-1] + timedelta(days=i) for i in range(1, pred_days+1)]

# total
total_days = index_days + next_days


# %% Banco de dados de predicao

CasosPre = np.concatenate((predictions, predictions_p))[:, 0]  # Total
CasosReais = df[n_steps:].values[:, 0].astype(int)
CasosDiasPre = pd.Series(CasosPre).diff()
CasosDias = pd.Series(CasosReais).diff()
CasosMM7 = round(CasosDiasPre.rolling(7).mean(), 2)
CasosMM14 = round(CasosDiasPre.rolling(14).mean(), 2)

predict = pd.DataFrame([total_days,
                        list(CasosPre),
                        list(CasosReais),
                        list(CasosDiasPre),
                        list(CasosDias),
                        list(CasosMM7),
                        list(CasosMM14)],
                       ["Data", "CasosPre", "CasosReais",
                        "CasosDiasPre", "CasosDias",
                        "CasosMM7", "CasosMM14"]).\
                        transpose().set_index("Data")


# %%  Gráfico de casos ou mortes reais e previstos do local selecionado

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
              go.Scatter(
                         x=predict.index,
                         y=predict['CasosPre'],
                         mode='lines+markers',
                         name=tipo + ' Previstos',
                         line_color='crimson'),
              secondary_y=True
              )

fig.add_trace(
              go.Scatter(
                          x=predict.index,
                          y=predict['CasosReais'],
                          mode='lines+markers',
                          name=tipo + ' Reais',
                          line_color=cor1),
              secondary_y=True
              )

fig.add_trace(
              go.Bar(x=predict.index,
                     y=predict['CasosDiasPre'],
                     name='Diario Previsto',
                     marker_color='tan'),
              secondary_y=False
              )

fig.add_trace(
              go.Bar(x=predict.index,
                     y=predict['CasosDias'],
                     name='Diario Real',
                     marker_color=cor2),
              secondary_y=False
              )

fig.add_trace(
              go.Scatter(x=predict.index,
                         y=predict['CasosMM7'],
                         name=tipo + ' MM7',
                         marker_color='black'),
              secondary_y=False
              )

fig.add_trace(
              go.Scatter(x=predict.index,
                         y=predict['CasosMM14'],
                         name=tipo + ' MM14',
                         marker_color='green'),
              secondary_y=False
              )

# Criando Layout
fig.update_layout(title_text=local + ' Previsão e Evolução de ' + tipo,
                  legend=dict(x=0.02, y=0.95), legend_orientation="v")

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text=tipo + " Total", secondary_y=True)
fig.update_yaxes(title_text=tipo + " Diarios", secondary_y=False)

graph = local + ' LSTM - Gráfico Previsão de ' + tipo + ' ' + day + '.html'
py.plot(fig, filename=graph)


# %%
