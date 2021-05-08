#!/usr/bin/python
# -*- coding: utf-8 -*-

## teste de leitura de arquivos nos formatos Python3
## uso da função 'locale' para formatar moeda brasileira

import locale


def formata_moeda(valor):
    return locale.currency(float(valor))

gastos = open('gastos.txt', 'r') ## Carrega o arquivo txt

gastos_float = map(float, gastos) ## Converte para float

for gasto in gastos:
    gasto.replace('.', ',')
    print(gasto)

