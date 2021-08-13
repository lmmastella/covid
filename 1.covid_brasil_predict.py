#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:28:28 2020

Baseado:

https://covid.saude.gov.br/

@author: lmmastella
"""

# %% import

import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# %% database

# df_brasil_ori = pd.read_excel('HIST_PAINEL_COVIDBR_25jun2020.xlsx')

df_brasil_ori = pd.read_csv(
    "/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR.csv", sep=";"
)
# limpar os campos sem dados
df_brasil_ori = df_brasil_ori.replace(np.nan, "", regex=True)

# %% pre-processamento datasets - pivot e inclusao da linha Brasil

df_brasil_ori = df_brasil_ori.replace(np.nan, '', regex=True)

df_brasil = df_brasil_ori[df_brasil_ori['codmun'] == ''].\
            drop(columns=['municipio', 'coduf', 'codmun',
                          'codRegiaoSaude', 'nomeRegiaoSaude',
                          'semanaEpi', 'populacaoTCU2019',
                          'Recuperadosnovos', 'emAcompanhamentoNovos'])

df_brasil.loc[df_brasil['estado'] == '', 'estado'] = 'BR'


df_casos = df_brasil.pivot(index='estado', columns='data',
                           values='casosAcumulado').reset_index().fillna(0)

df_mortes = df_brasil.pivot(index='estado', columns='data',
                            values='obitosAcumulado').reset_index().fillna(0)

# determinar o ultimo dia de analise e o dataset
if all(df_casos.iloc[:, -1:].all() != 0):
    dia_atual = df_casos.columns[-1]
else:
    dia_atual = df_casos.columns[-2]

# %% variaveis

# estados
estado = 'RS'
n_dias = 100
n_casos = 10
n_estados = 10

# %% escolha dos estados a serem analisados

estados = df_casos[df_casos[dia_atual] >= n_casos].\
          sort_values(by=dia_atual, ascending=False)
estados = estados['estado'].reset_index(drop=True).loc[:n_estados].tolist()


# %% analise de incremento de casos e mortes

def plotEstado(estado):
    """
    Parameters
    ----------

    estado : estados a serem analisados

    Returns
    -------
    descricao de cada pais com casos e mortes

    """

    # variaveis
    # firtsdeath = mortes_inc.index[0]
    # totaldays_m = mortes_inc.size
    # current_m = int(mortes_inc.iloc[-1])
    # lastweek_m = int(mortes_inc.iloc[-8])
    # today_inc_m = float(mortes_inc_day.iloc[-1])
    # week_inc_m = float(mortes_inc_week.iloc[-1])

    # variaveis
    # firtscase = casos_inc.index[0]
    # totaldays_c = casos_inc.size
    # current_c = int(casos_inc.iloc[-1])
    # lastweek_c = int(casos_inc.iloc[-8])
    # today_inc_c = float(casos_inc_day.iloc[-1])
    # week_inc_c = float(casos_inc_week.iloc[-1])

    # resumo casos
    casos_inc = df_casos[df_casos['estado'] == estado].iloc[:, 1:].T.sum(axis=1)
    casos_inc = pd.DataFrame(casos_inc)
    casos_inc.columns = ['Casos']
    casos_inc = casos_inc.loc[casos_inc['Casos'] > 0]

    # incrementos
    casos_inc_day = round(100*casos_inc.pct_change(), 2)
    casos_inc_week = round(100*casos_inc.pct_change(periods=7), 2)

    # resumo mortes
    mortes_inc = df_mortes[df_mortes['estado'] == estado].iloc[:, 4:].T.sum(axis=1)
    mortes_inc = pd.DataFrame(mortes_inc)
    mortes_inc.columns = ['Mortes']
    mortes_inc = mortes_inc.loc[mortes_inc['Mortes'] > 0]

    # incrementos
    mortes_inc_day = round(100*mortes_inc.pct_change(), 2)
    mortes_inc_week = round(100*mortes_inc.pct_change(periods=7), 2)

    # progressao de casos
    print('\n** Based on Most Recent Week of Data **\n')
    print('\tFirst case on', casos_inc.index[0], '\t', '\t', '\t',
          casos_inc.size, 'days after')
    print('\tConfirmed cases on', casos_inc.index[-1], '\t',
          int(casos_inc.iloc[-1]))
    print('\tConfirmed cases on', casos_inc.index[-8], '\t',
          int(casos_inc.iloc[-8]))
    ratio_w = int(casos_inc.iloc[-1])/int(casos_inc.iloc[-8])
    print('\tRatio Weekly:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_w, 2))
    print('\tWeekly increase:', '\t', '\t', '\t', '\t',
          round(float(casos_inc_week.iloc[-1]), 2), '%')
    ratio_d = int(casos_inc.iloc[-1])/int(casos_inc.iloc[-2])
    print('\tRatio Daily:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_d, 2))
    print('\tDaily increase:', '\t', '\t', '\t', '\t', '\t',
          round(float(casos_inc_day.iloc[-1]), 2), '% per day')
    recentdbltime = round(7 * np.log(2) / np.log(ratio_w), 1)
    print('\tDoubling Time (represents recent growth):',
          recentdbltime, 'days')

    print()

    # progressao de mortes
    print('\tFirst death on', mortes_inc.index[0], '\t', '\t',
          mortes_inc.size, 'days after')
    print('\tConfirmed deaths on', mortes_inc.index[-1], '\t',
          int(mortes_inc.iloc[-1]))
    print('\tConfirmed deaths on', mortes_inc.index[-8], '\t',
          int(mortes_inc.iloc[-8]))
    ratio_wm = int(mortes_inc.iloc[-1])/int(mortes_inc.iloc[-8])
    print('\tRatio Weekly:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_wm, 2))
    print('\tWeekly increase:', '\t', '\t', '\t', '\t',
          round(float(mortes_inc_week.iloc[-1]), 2), '%')
    ratio_dm = int(mortes_inc.iloc[-1])/int(mortes_inc.iloc[-2])
    print('\tRatio Daily:', '\t', '\t', '\t', '\t', '\t',
          round(ratio_dm, 2))
    print('\tDaily increase:', '\t', '\t', '\t', '\t', '\t',
          round(float(mortes_inc_day.iloc[-1]), 2), '% per day')
    recentdbltime = round(7 * np.log(2) / np.log(ratio_wm), 1)
    print('\tDoubling Time (represents recent growth):',
          recentdbltime, 'days')
    print('\tTaxa de Mortalidade: ', round(100*int(mortes_inc.iloc[-1])
                                           / int(casos_inc.iloc[-1]), 2), '%')
    print()


# %% solicitar analise

for e in estados:
    print()
    print(e)
    plotEstado(e)

# %% dados para gráficos 1, 2, 3 e 4 REPETIDO > func
# evolução casos x mortes no estado selecionado

# resumo casos
casos_inc = df_casos[df_casos['estado'] == estado].iloc[:, 1:].T.sum(axis=1)
casos_inc = pd.DataFrame(casos_inc)
casos_inc.columns = ['Casos']
casos_inc = casos_inc.loc[casos_inc['Casos'] > 0]
casos_day = casos_inc.diff()

# incrementos
casos_inc_day = round(100*casos_inc.pct_change(), 2)
casos_inc_week = round(100*casos_inc.pct_change(periods=7), 2)

# resumo mortes
mortes_inc = df_mortes[df_mortes['estado'] == estado].iloc[:, 4:].T.sum(axis=1)
mortes_inc = pd.DataFrame(mortes_inc)
mortes_inc.columns = ['Mortes']
mortes_inc = mortes_inc.loc[mortes_inc['Mortes'] > 0]
mortes_day = mortes_inc.diff()

# incrementos
mortes_inc_day = round(100*mortes_inc.pct_change(), 2)
mortes_inc_week = round(100*mortes_inc.pct_change(periods=7), 2)

# %% 1.1 grafico de casos e mortes no estado selecionado

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
layout = go.Layout(title=estado + '  Evolução de casos e mortes',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)

# exibindo figura/gráfico
name = estado+'_casos_mortes.html'
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
fig.update_layout(title_text=estado + '  Evolução de casos')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos Diarios", secondary_y=False)

# fig.show()
# exibindo figura/gráfico
name = estado + ' Evolucao de casos_diarios.html'
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
fig.update_layout(title_text=estado + '  Evolução de mortes')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Mortes Total", secondary_y=True)
fig.update_yaxes(title_text="Mortes Diarias", secondary_y=False)

# fig.show()
# exibindo figura/gráfico
name = estado + ' Evolucao de mortes diarias.html'
py.plot(fig, filename=name)

# %% 2 grafico de evolução em % de casos semanais e diarios no estado selecionado

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
layout = go.Layout(title=estado + '  Evolução de casos em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = estado + '_evolução_casos.html'
py.plot(fig, filename=name)


# %% 3 grafico de evolução de mortes semanais e diarios no estado selecionado

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
layout = go.Layout(title=estado + '  Evolução de mortes em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = estado + '_evolução_mortes.html'
py.plot(fig, filename=name)

# %% 4 grafico total de evolução de casos e mortes diarios e semanais no estado

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
layout = go.Layout(title=estado + '  Evolução de casos e morte em %',
                   yaxis={'title': '%'},
                   xaxis={'title': 'Data'})

# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)


# exibindo figura/gráfico
name = estado + '_evolução_total.html'
py.plot(fig, filename=name)

# %% processamento da dataset com estados escolhidos

cas = pd.DataFrame()

for e in estados:
    casos = df_casos[df_casos['estado'] == e].iloc[:, 4:].T.sum(axis=1)
    casos = pd.DataFrame(casos)
    cas = pd.concat([cas, casos], ignore_index=True, axis=1)

cas.columns = estados

# %% 5 Grafico geral


p0 = go.Scatter(x=cas.index, y=cas[estados[0]],
                mode='lines', name=estados[0])
p1 = go.Scatter(x=cas.index, y=cas[estados[1]],
                mode='lines', name=estados[1])
p2 = go.Scatter(x=cas.index, y=cas[estados[2]],
                mode='lines', name=estados[2])
p3 = go.Scatter(x=cas.index, y=cas[estados[3]],
                mode='lines', name=estados[3])
p4 = go.Scatter(x=cas.index, y=cas[estados[4]],
                mode='lines', name=estados[4])
p5 = go.Scatter(x=cas.index, y=cas[estados[5]],
                mode='lines', name=estados[5])
p6 = go.Scatter(x=cas.index, y=cas[estados[6]],
                mode='lines', name=estados[6])
p7 = go.Scatter(x=cas.index, y=cas[estados[7]],
                mode='lines', name=estados[7])
p8 = go.Scatter(x=cas.index, y=cas[estados[8]],
                mode='lines', name=estados[8])
p9 = go.Scatter(x=cas.index, y=cas[estados[9]],
                mode='lines', name=estados[9])
p10 = go.Scatter(x=cas.index, y=cas[estados[10]],
                 mode='lines', name=estados[10])
p11 = go.Scatter(x=cas.index, y=cas[estados[11]],
                 mode='lines', name=estados[11])
p12 = go.Scatter(x=cas.index, y=cas[estados[12]],
                 mode='lines', name=estados[12])
p13 = go.Scatter(x=cas.index, y=cas[estados[13]],
                 mode='lines', name=estados[13])
p14 = go.Scatter(x=cas.index, y=cas[estados[14]],
                 mode='lines', name=estados[14])

data = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

layout = go.Layout(title=' Evolução de casos COVID-19',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Data'})

fig = go.Figure(data=data, layout=layout)

# py.iplot(fig)
name = 'Estados Total.html'
py.plot(fig, filename=name)


# %% 6 dados para analise entre os estado desde primeiro caso

cas_0 = pd.DataFrame()

for e in estados:
    casos_0 = df_casos[df_casos['estado'] == e].iloc[:, 4:].T.sum(axis=1)
    casos_0 = pd.DataFrame(casos_0)
    casos_0.columns = [e]
    casos_0 = casos_0.loc[casos_0[e] > 0].iloc[0:n_dias]
    casos_0 = casos_0.reset_index(drop=True)
    cas_0 = pd.concat([cas_0, casos_0], ignore_index=True, axis=1)

cas_0.columns = estados

# %% 6 grafico total (dias) - plotly

p0 = go.Scatter(x=cas_0.index, y=cas_0[estados[0]],
                mode='lines', name=estados[0])
p1 = go.Scatter(x=cas_0.index, y=cas_0[estados[1]],
                mode='lines', name=estados[1])
p2 = go.Scatter(x=cas_0.index, y=cas_0[estados[2]],
                mode='lines', name=estados[2])
p3 = go.Scatter(x=cas_0.index, y=cas_0[estados[3]],
                mode='lines', name=estados[3])
p4 = go.Scatter(x=cas_0.index, y=cas_0[estados[4]],
                mode='lines', name=estados[4])
p5 = go.Scatter(x=cas_0.index, y=cas_0[estados[5]],
                mode='lines', name=estados[5])
p6 = go.Scatter(x=cas_0.index, y=cas_0[estados[6]],
                mode='lines', name=estados[6])
p7 = go.Scatter(x=cas_0.index, y=cas_0[estados[7]],
                mode='lines', name=estados[7])
p8 = go.Scatter(x=cas_0.index, y=cas_0[estados[8]],
                mode='lines', name=estados[8])
p9 = go.Scatter(x=cas_0.index, y=cas_0[estados[9]],
                mode='lines', name=estados[9])
p10 = go.Scatter(x=cas_0.index, y=cas_0[estados[10]],
                 mode='lines', name=estados[10])
p11 = go.Scatter(x=cas_0.index, y=cas_0[estados[11]],
                 mode='lines', name=estados[11])
p12 = go.Scatter(x=cas_0.index, y=cas_0[estados[12]],
                 mode='lines', name=estados[12])
p13 = go.Scatter(x=cas_0.index, y=cas_0[estados[13]],
                 mode='lines', name=estados[13])
p14 = go.Scatter(x=cas_0.index, y=cas_0[estados[14]],
                 mode='lines', name=estados[14])


data = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

layout = go.Layout(title='Evolução de ' + str(n_dias) + ' dias da COVID-19 desde primeiro caso',
                   yaxis={'title': 'Casos'},
                   xaxis={'title': 'Dias'})

fig = go.Figure(data=data, layout=layout)

name = 'Estados Caso_0_40.html'
py.plot(fig, filename=name)


# %% Histogramas

for e in estados:
    cas_cas = df_casos[df_casos['estado'] == e].iloc[:, 4:].T.sum(axis=1)
    cas_cas = pd.DataFrame(cas_cas)
    cas_cas.columns = [e]
    cas_cas = cas_cas.loc[cas_cas[e] > 0]
    cas_cas = cas_cas.reset_index(drop=True).diff()
    print(e, '\n', '\t', 'Maximo de casos: ' + str(int(cas_cas[e].max())) +
          '\t', '\t' 'no dia: ' + str(cas_cas[e].idxmax()) +
          ' total de dias: ' + str(cas_cas[e].index[-1]+1))
    fig = px.bar(cas_cas, x=cas_cas.index, y=cas_cas[e])
    fig.show()
    # name = e + ' Histograma.html'
    # py.plot(fig, filename=name)
    # teste
