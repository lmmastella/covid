#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:28:28 2020

Baseado:
Canal Sandeco
https://www.youtube.com/watch?v=VepbhWm9E5M

Repository by Johns Hopkins CSSE
https://github.com/CSSEGISandData/COVID-19

@author: lmmastella
"""

# %% import

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from datetime import datetime, timedelta
import os
import warnings

# %% database

df_casos = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_mortes = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
# df_recuperados = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


# %% variaveis

pais = 'Brazil'
n_dias = 70
n_casos = 20000
n_paises = 15
dia_atual = df_casos.columns[-1]

# %% escolha dos paises a serem analisados

paises = df_casos[df_casos[dia_atual] >= n_casos].\
         sort_values(by=dia_atual, ascending=False)
paises = paises['Country/Region'].reset_index(drop=True)
paises = paises.iloc[:n_paises].tolist()

# %% pre-processamento da dataset com paises escolhidos

cas = pd.DataFrame()

for p in paises:
    casos = df_casos[df_casos['Country/Region'] == p].iloc[:, 4:].T.sum(axis=1)
    casos = pd.DataFrame(casos)
    cas = pd.concat([cas, casos], ignore_index=True, axis=1)

cas.columns = paises


# %% funcao de analise de incremento de casos e mortes


def plotCountry(country):
    """
    Parameters
    ----------

    country : paises a serem analisados

    Returns
    -------
    descricao de cada pais com casos e mortes

    """
    # resumo casos
    casos_inc = df_casos[df_casos['Country/Region'] == country].iloc[:, 4:].T.sum(axis=1)
    casos_inc = pd.DataFrame(casos_inc)
    casos_inc.columns = ['Casos']
    casos_inc = casos_inc.loc[casos_inc['Casos'] > 0]

    # incrementos
    casos_inc_day = casos_inc.pct_change()
    casos_inc_week = casos_inc.pct_change(periods=7)

    # variaveis
    # firtscase = casos_inc.index[0]
    # totaldays_c = casos_inc.size
    # current_c = int(casos_inc.iloc[-1])
    # lastweek_c = int(casos_inc.iloc[-8])
    # today_inc_c = float(casos_inc_day.iloc[-1])
    # week_inc_c = float(casos_inc_week.iloc[-1])

    # resumo mortes
    mortes_inc = df_mortes[df_mortes['Country/Region'] == country].iloc[:, 4:].T.sum(axis=1)
    mortes_inc = pd.DataFrame(mortes_inc)
    mortes_inc.columns = ['Mortes']
    mortes_inc = mortes_inc.loc[mortes_inc['Mortes'] > 0]

    # incrementos
    mortes_inc_day = mortes_inc.pct_change()
    mortes_inc_week = mortes_inc.pct_change(periods=7)

    # variaveis
    # firtsdeath = mortes_inc.index[0]
    # totaldays_m = mortes_inc.size
    # current_m = int(mortes_inc.iloc[-1])
    # lastweek_m = int(mortes_inc.iloc[-8])
    # today_inc_m = float(mortes_inc_day.iloc[-1])
    # week_inc_m = float(mortes_inc_week.iloc[-1])

    # progressao de casos
    print('\n** Based on Most Recent Week of Data **\n')
    print('\tFirst case on', casos_inc.index[0], '\t', '\t', '\t',
          casos_inc.size, 'days after')
    print('\tConfirmed cases on', casos_inc.index[-1], '\t', '\t',
          int(casos_inc.iloc[-1]))
    print('\tConfirmed cases on', casos_inc.index[-8], '\t', '\t',
          int(casos_inc.iloc[-8]))
    ratio_w = int(casos_inc.iloc[-1])/int(casos_inc.iloc[-8])
    print('\tRatio Weekly:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_w, 2))
    print('\tWeekly increase:', '\t', '\t', '\t', '\t',
          round(100*float(casos_inc_week.iloc[-1]), 2), '%')
    ratio_d = int(casos_inc.iloc[-1])/int(casos_inc.iloc[-2])
    print('\tRatio Daily:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_d, 2))
    print('\tDaily increase:', '\t', '\t', '\t', '\t', '\t',
          round(100*float(casos_inc_day.iloc[-1]), 2), '% per day')
    recentdbltime = round(7 * np.log(2) / np.log(ratio_w), 1)
    print('\tDoubling Time (represents recent growth):',
          recentdbltime, 'days')

    print()

    # progressao de mortes
    print('\tFirst death on', mortes_inc.index[0], '\t', '\t', '\t',
          mortes_inc.size, 'days after')
    print('\tConfirmed deaths on', mortes_inc.index[-1], '\t', '\t',
          int(mortes_inc.iloc[-1]))
    print('\tConfirmed deaths on', mortes_inc.index[-8], '\t', '\t',
          int(mortes_inc.iloc[-8]))
    ratio_wm = int(mortes_inc.iloc[-1])/int(mortes_inc.iloc[-8])
    print('\tRatio Weekly:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_wm, 2))
    print('\tWeekly increase:', '\t', '\t', '\t', '\t',
          round(100*float(mortes_inc_week.iloc[-1]), 2), '%')
    ratio_dm = int(mortes_inc.iloc[-1])/int(mortes_inc.iloc[-2])
    print('\tRatio Daily:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_dm, 2))
    print('\tDaily increase:', '\t', '\t', '\t', '\t', '\t',
          round(100*float(mortes_inc_day.iloc[-1]), 2), '% per day')
    recentdbltime = round(7 * np.log(2) / np.log(ratio_wm), 1)
    print('\tDoubling Time (represents recent growth):',
          recentdbltime, 'days')
    print('\tTaxa de Mortalidade: ', round(100*int(mortes_inc.iloc[-1])
                                           / int(casos_inc.iloc[-1]), 2), '%')
    print()


# %% solicitar analise da funcao

print()
print('Mundial', '\t', '\t', '\t', '\t', dia_atual)
print()
print('\tConfirmed cases ', '\t', df_casos[dia_atual].sum())
print('\tConfirmed deaths ', '\t', df_mortes[dia_atual].sum())

for p in paises:
    print(p)
    plotCountry(p)

# =============================================================================
#
#  Dados e Graficos para  evolução de casos e mortes
#
# =============================================================================

# %% dados para gráficos 1, 2, 3 e 4 evolução casos x mortes por pais


casos_inc = df_casos[df_casos['Country/Region'] == pais].iloc[:, 4:].T.sum(axis=1)
casos_inc = pd.DataFrame(casos_inc)
casos_inc.columns = ['Casos']
casos_inc = casos_inc.loc[casos_inc['Casos'] > 0]
casos_day = casos_inc.diff()

# incrementos
casos_inc_day = round(100*casos_inc.pct_change(), 2)
casos_inc_week = round(100*casos_inc.pct_change(periods=7), 2)

# resumo mortes
mortes_inc = df_mortes[df_mortes['Country/Region'] == pais].iloc[:, 4:].T.sum(axis=1)
mortes_inc = pd.DataFrame(mortes_inc)
mortes_inc.columns = ['Mortes']
mortes_inc = mortes_inc.loc[mortes_inc['Mortes'] > 0]
mortes_day = mortes_inc.diff()

# incrementos
mortes_inc_day = round(100*mortes_inc.pct_change(), 2)
mortes_inc_week = round(100*mortes_inc.pct_change(periods=7), 2)

# %% 1.1 grafico de casos e mortes totais por pais

p1 = go.Scatter(x=casos_inc.index,
                y=casos_inc['Casos'],
                mode='lines',
                name='Casos')

p2 = go.Scatter(x=mortes_inc.index,
                y=mortes_inc['Mortes'],
                mode='lines',
                name='Mortes')
# Armazenando gráfico em uma lista
data = [p1, p2]

# Criando Layout
layout = go.Layout(title=pais + '  Evolução de casos e mortes',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)

# exibindo figura/gráfico
name = pais+'_casos_mortes.html'
py.plot(fig, filename=name)

# %% 1.2 grafico de casos no estado selecionado

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
              go.Scatter(
                         x=casos_inc.index,
                         y=casos_inc['Casos'],
                         mode='lines',
                         name='Total'),
              secondary_y=True
              )

fig.add_trace(
              go.Bar(x=casos_day.index,
                     y=casos_day['Casos'],
                     name='Diario'),
              secondary_y=False
              )

# Criando Layout
fig.update_layout(title_text=pais + '  Evolução de casos')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos Diarios", secondary_y=False)

# fig.show()
# exibindo figura/gráfico
name = pais + ' Evolucao de casos_diarios.html'
py.plot(fig, filename=name)

# %% 1.3 grafico de mortes no estado selecionado

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
              go.Scatter(
                         x=mortes_inc.index,
                         y=mortes_inc['Mortes'],
                         mode='lines',
                         name='Total'),
              secondary_y=True
              )

fig.add_trace(
              go.Bar(x=mortes_day.index,
                     y=mortes_day['Mortes'],
                     name='Diario'),
              secondary_y=False
              )

# Criando Layout
fig.update_layout(title_text=pais + '  Evolução de mortes')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Mortes Total", secondary_y=True)
fig.update_yaxes(title_text="Mortes Diarias", secondary_y=False)

# fig.show()
# exibindo figura/gráfico
name = pais + ' Evolucao de mortes diarias.html'
py.plot(fig, filename=name)

# %% 2 grafico de evolução em % de casos semanais e diarios por pais

p1 = go.Scatter(x=casos_inc_day.index,
                y=casos_inc_day['Casos'],
                mode='lines',
                name='Diario')

p2 = go.Scatter(x=casos_inc_week.index,
                y=casos_inc_week['Casos'],
                mode='lines',
                name='Semanal')
# Armazenando gráfico em uma lista
data = [p1, p2]

# Criando Layout
layout = go.Layout(title=pais + '  Evolução de casos em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = pais + '_evolução_casos.html'
py.plot(fig, filename=name)


# %% 3 grafico de evolução de mortes semanais e diarios por pais

p1 = go.Scatter(x=mortes_inc_day.index,
                y=mortes_inc_day['Mortes'],
                mode='lines',
                name='Diario')

p2 = go.Scatter(x=mortes_inc_week.index,
                y=mortes_inc_week['Mortes'],
                mode='lines',
                name='Semanal')
# Armazenando gráfico em uma lista
data = [p1, p2]

# Criando Layout
layout = go.Layout(title=pais + '  Evolução de mortes em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = pais + '_evolução_mortes.html'
py.plot(fig, filename=name)

# %% 4 grafico total de evolução de casos e mortes diarios e semanais

casos_day = go.Scatter(x=casos_inc_day.index,
                       y=casos_inc_day['Casos'],
                       mode='lines',
                       name='Casos Diarios')

casos_week = go.Scatter(x=casos_inc_week.index,
                        y=casos_inc_week['Casos'],
                        mode='lines',
                        name='Casos Semanais')
mortes_day = go.Scatter(x=mortes_inc_day.index,
                        y=mortes_inc_day['Mortes'],
                        mode='lines',
                        name='Mortes Diarios')

mortes_week = go.Scatter(x=mortes_inc_week.index,
                         y=mortes_inc_week['Mortes'],
                         mode='lines',
                         name='Mortes Semanais')
# Armazenando gráfico em uma lista
data = [casos_day, casos_week, mortes_day, mortes_week]

# Criando Layout
layout = go.Layout(title=pais + '  Evolução de casos e morte em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = pais + '_evolução_total.html'
py.plot(fig, filename=name)


# %% 5 Grafico geral


p0 = go.Scatter(x=cas.index, y=cas[paises[0]],
                mode='lines', name=paises[0])
p1 = go.Scatter(x=cas.index, y=cas[paises[1]],
                mode='lines', name=paises[1])
p2 = go.Scatter(x=cas.index, y=cas[paises[2]],
                mode='lines', name=paises[2])
p3 = go.Scatter(x=cas.index, y=cas[paises[3]],
                mode='lines', name=paises[3])
p4 = go.Scatter(x=cas.index, y=cas[paises[4]],
                mode='lines', name=paises[4])
p5 = go.Scatter(x=cas.index, y=cas[paises[5]],
                mode='lines', name=paises[5])
p6 = go.Scatter(x=cas.index, y=cas[paises[6]],
                mode='lines', name=paises[6])
p7 = go.Scatter(x=cas.index, y=cas[paises[7]],
                mode='lines', name=paises[7])
p8 = go.Scatter(x=cas.index, y=cas[paises[8]],
                mode='lines', name=paises[8])
p9 = go.Scatter(x=cas.index, y=cas[paises[9]],
                mode='lines', name=paises[9])
p10 = go.Scatter(x=cas.index, y=cas[paises[10]],
                 mode='lines', name=paises[10])
p11 = go.Scatter(x=cas.index, y=cas[paises[11]],
                 mode='lines', name=paises[11])
p12 = go.Scatter(x=cas.index, y=cas[paises[12]],
                 mode='lines', name=paises[12])
p13 = go.Scatter(x=cas.index, y=cas[paises[13]],
                 mode='lines', name=paises[13])
p14 = go.Scatter(x=cas.index, y=cas[paises[14]],
                 mode='lines', name=paises[14])

data = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

layout = go.Layout(title=' Evolução de casos COVID-19',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Data'})

fig = go.Figure(data=data, layout=layout)

# py.iplot(fig)
name = pais + ' Total.html'
py.plot(fig, filename=name)


# %% 6 dados para analise entre os paises desde primeiro caso

cas_0 = pd.DataFrame()

for p in paises:
    casos_0 = df_casos[df_casos['Country/Region'] == p].iloc[:, 4:].T.sum(axis=1)
    casos_0 = pd.DataFrame(casos_0)
    casos_0.columns = [p]
    casos_0 = casos_0.loc[casos_0[p] > 0]
#   casos_0 = casos_0.loc[casos_0[p] > 0].iloc[0:n_dias]
    casos_0 = casos_0.reset_index(drop=True)
    cas_0 = pd.concat([cas_0, casos_0], ignore_index=True, axis=1)

cas_0.columns = paises


# %% 6 grafico total (dias) - plotly

p0 = go.Scatter(x=cas_0.index, y=cas_0[paises[0]],
                mode='lines', name=paises[0])
p1 = go.Scatter(x=cas_0.index, y=cas_0[paises[1]],
                mode='lines', name=paises[1])
p2 = go.Scatter(x=cas_0.index, y=cas_0[paises[2]],
                mode='lines', name=paises[2])
p3 = go.Scatter(x=cas_0.index, y=cas_0[paises[3]],
                mode='lines', name=paises[3])
p4 = go.Scatter(x=cas_0.index, y=cas_0[paises[4]],
                mode='lines', name=paises[4])
p5 = go.Scatter(x=cas_0.index, y=cas_0[paises[5]],
                mode='lines', name=paises[5])
p6 = go.Scatter(x=cas_0.index, y=cas_0[paises[6]],
                mode='lines', name=paises[6])
p7 = go.Scatter(x=cas_0.index, y=cas_0[paises[7]],
                mode='lines', name=paises[7])
p8 = go.Scatter(x=cas_0.index, y=cas_0[paises[8]],
                mode='lines', name=paises[8])
p9 = go.Scatter(x=cas_0.index, y=cas_0[paises[9]],
                mode='lines', name=paises[9])
p10 = go.Scatter(x=cas_0.index, y=cas_0[paises[10]],
                 mode='lines', name=paises[10])
p11 = go.Scatter(x=cas_0.index, y=cas_0[paises[11]],
                 mode='lines', name=paises[11])
p12 = go.Scatter(x=cas_0.index, y=cas_0[paises[12]],
                 mode='lines', name=paises[12])
p13 = go.Scatter(x=cas_0.index, y=cas_0[paises[13]],
                 mode='lines', name=paises[13])
p14 = go.Scatter(x=cas_0.index, y=cas_0[paises[14]],
                 mode='lines', name=paises[14])


data = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

layout = go.Layout(title='Evolução de ' + str(n_dias) + ' dias da COVID-19 desde primeiro caso',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Dias'})

fig = go.Figure(data=data, layout=layout)

name = pais + ' Caso_0_40.html'
py.plot(fig, filename=name)

# %% Histogramas

for p in paises:
    cas_cas = df_casos[df_casos['Country/Region'] == p].iloc[:, 4:].T.sum(axis=1)
    cas_cas = pd.DataFrame(cas_cas)
    cas_cas.columns = [p]
    cas_cas = cas_cas.loc[cas_cas[p] > 0]
    cas_cas = cas_cas.reset_index(drop=True).diff()
    print(p, '\n', '\t', 'Maximo de casos: ' + str(int(cas_cas[p].max())) +
          '\t', '\t', 'no dia: ' + str(cas_cas[p].idxmax()) +
          ' total de dias: ' + str(cas_cas[p].index[-1]+1))
    fig = px.bar(cas_cas, x=cas_cas.index, y=cas_cas[p])
    fig.show()
    # name = p + ' Histograma.html'
    # py.plot(fig, filename=name)

# %% Predict Variaveis Mundial

# estados
tipo = 'Casos'  # 'Casos' ou 'Mortes'
estado = 'France'
n_dias = 60
pred_dias = 10

# %% Casos - Prediction Curve for Global Cases


if tipo == 'Mortes':

    df = df_mortes[df_mortes['Country/Region'] == estado].iloc[:, 4:].T.sum(axis=1)
    cor1 = 'blue'
    cor2 = 'red'

else:

    df = df_casos[df_casos['Country/Region'] == estado].iloc[:, 4:].T.sum(axis=1)
    cor1 = 'blue'
    cor2 = 'royalblue'


df = pd.DataFrame(df)
df.columns = [tipo]
df = df.loc[df[tipo] > 0]

data_y = np.log10(np.asarray(df)).astype('float64')
data_x = np.arange(1, len(data_y)+1).reshape([data_y.shape[0], 1])


# %% Casos - model

Visible = Input(shape=(1,))
Dense_l1 = Dense(80, name="Dense_l1")(Visible)
LRelu_l1 = LeakyReLU(name="LRelu_l1")(Dense_l1)

Dense_l2 = Dense(80, name="Dense_l2")(LRelu_l1)
LRelu_l2 = LeakyReLU(name="LRelu_l2")(Dense_l2)

Dense_l3 = Dense(80, name="Dense_l3")(LRelu_l2)
LRelu_l3 = LeakyReLU(name="LRelu_l3")(Dense_l3)

Dense_l4 = Dense(1, name="Dense_l4")(LRelu_l3)
LRelu_l4 = LeakyReLU(name="Output")(Dense_l4)

model = models.Model(inputs=Visible, outputs=LRelu_l4)
model.compile(optimizer=Adam(lr=0.001),
              loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()


# %% Casos - training

epochs = 10000
arq = 'Covid_' + tipo + '_' + estado + '.h5'

mcp = ModelCheckpoint(filepath=arq, monitor='loss',
                      save_best_only=True, verbose=1)

if(not os.path.exists(arq)):
    model.fit(data_x, data_y,
              epochs=epochs,
              callbacks=[mcp])
    model.save(arq)

model = load_model(arq)

model.summary()
loss, metric = model.evaluate(data_x, data_y)

# =============================================================================
# loss, metric = model.evaluate(data_x, data_y)
# Casos 80/80 [========================] - 0s 76us/sample - loss: 2.0755e-04
# Mortes 60/60 [=======================] - 0s 55us/sample - loss: 1.2554e-04
# =============================================================================

# %% Casos - text prediction casos confirmed

dias = [datetime.strptime(d, '%m/%d/%y').strftime("%d %b")
        for d in df.index]

nextdays = [(datetime.strptime(dias[-1], '%d %b') +
             timedelta(days=i)).strftime("%d %b")
            for i in range(1, pred_dias+1)]

total = dias + nextdays

df_pre = np.power(10, model.predict(np.arange(1, len(df) +
                                              pred_dias+1))).astype(int)

text = 'Predições de ' + tipo + ':' + '\n'

for i in range(pred_dias):
    text += nextdays[i]+" : "+str(np.round(df_pre[-1*(pred_dias-i)])[0]) + "\n"

# %% Casos - tabulation of prediction casos and actual figure after

predict = pd.DataFrame([total[-n_dias:],
                        list(np.int64(np.round(df_pre[-n_dias:].reshape(-1)))),
                        list(df[tipo][-n_dias+pred_dias:]),
                        list(np.int64(np.round(np.diff(df_pre[-n_dias-1:].reshape(-1))))),
                        list(df[tipo].diff()[-n_dias+pred_dias:])],
                       ["Data", "CasosPre", "CasosReais", "CasosDiasPre", "CasosDias"]).\
                         transpose().set_index("Data")

# %%  Casos - grafico de casos ou mortes reais e previstos no estado selecionado

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

# Criando Layout
fig.update_layout(title_text=estado + ' Previsão e Evolução de ' + tipo,
                  legend=dict(x=0.02, y=0.95), legend_orientation="v")

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text=tipo + " Total", secondary_y=True)
fig.update_yaxes(title_text=tipo + " Diarios", secondary_y=False)

name = estado + ' Evolucao e Previsao de ' + tipo + '.html'
py.plot(fig, filename=name)

