#!/usr/bin/env python

# coding: utf-8

#------------------------------------------------------------------------------------------

# Дано: файлы с1.txt и c2.txt, содержащие клики пользователей со сплита 1 и 2 соответственно

#------------------------------------------------------------------------------------------

# Формат: каждая строка соответствует одной сессии и содержит десять цифр: нулей или
# единиц, обозначающих был ли клик в документ или нет в порядке возрастания ранга
# документа в поисковой выдаче.
# Например, начало с1.txt:
# 1 1 1 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0 0 0 0
# Означает следующее. Представлена информация о трех поисковых сессиях. Сессия 1 –
# пользователь просмотрел (кликнул) в документы 1, 2, 3. В сессию 2 кликов не было, а в
# сессии 3 был клик только в первый документ. 

#------------------------------------------------------------------------------------------

# Задача: методом бутстрепинга проверить гипотезу H0 о том, что CTR на двух сплитах можно
# объяснить шумами. Выполнить данный подсчет для любой другой кликовой метрики,
# например: количество отказов (доля сессий без кликов), доля кликов в первый результат,
# позиция последнего клика, среднее количество кликов на сессию и т.д.

#------------------------------------------------------------------------------------------

# Как считать метрику: Обычно кликовые метрики считают следующим образом: выбирают
# за определенный промежуток времени сессии (например за 1 минуту) и считают метрику.
# Например, за 1 минуту на сплит пришло 100 запросов, из них 20 сессий без кликов, тогда
# CTR равен (100-20)/100 = 0,8. В вашем случае информации о времени нет, поэтому возможно
# просто разбить сессии на группы по 100 или по 1000 сессий, превратить каждую группу в
# точки метрики, CTR или другую. Количество сессий в одном сплите 1 000 000, тогда взяв в
# группу, например, 1000 сессий, получим (1 000 000/1000) = 1000 значений CTR.
# Как проверять гипотезу: Считать t-статистику, и проверять условие t > t*, где t* = 1,96 для
# α=5%.
# Что делать с бутстрепингом: На каждом шаге семплированием получать новое множество
# значений метрики (аналогично, как делали на семинаре для множества побед и ничьих),
# проверять гипотезу и считать количество итераций, на которых гипотеза H0 может быть
# отвергнута. По результату посчитать ASL(уровень значимости) как count/N, где count —
# количество итераций при которых гипотезу H0 была отвергнута, а N общее количество
# итераций.

# В этой статье доступно описана схема эксперимента на двух сплитах и как считать t
# статистику
# http://www.exp-platform.com/Documents/GuideControlledExperiments.pdf

# Про бутстрепинг можно почитать
# http://goanna.cs.rmit.edu.au/~aht/tiger/p525-sakai.pdf
# http://www.stat.cmu.edu/~cshalizi/402/lectures/08-bootstrap/lecture-08.pdf

#------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

#------------------------------------------------------------------------------------------

c1 = pd.read_csv("./c1.txt", sep=" ", header=None)
c2 = pd.read_csv("./c2.txt", sep=" ", header=None)

#------------------------------------------------------------------------------------------

compute_ctr = lambda c: (c.sum(axis=1) > 0).as_matrix().reshape((len(c1) / 1000), 1000).mean(axis=1)
compute_last_click_pos = lambda c: (c1 * c1.columns).max(axis=1).as_matrix().reshape((len(c1) / 1000), 1000).mean(axis=1)

#------------------------------------------------------------------------------------------

ctr1 = compute_ctr(c1)
ctr2 = compute_ctr(c2)

last1 = compute_last_click_pos(c1)
last2 = compute_last_click_pos(c2)

#------------------------------------------------------------------------------------------


def t_test(x1, x2, alpha=0.05, verbose=False):
    x1_n = float(x1.shape[0])
    x2_n = float(x2.shape[0])
    x1_E = np.mean(x1)
    x2_E = np.mean(x2)
    x1_S = np.var(x1, ddof=1)
    x2_S = np.var(x2, ddof=1)
    S = ((x1_n - 1) * x1_S + (x2_n - 1) * x2_S) / (x1_n + x2_n - 2)
    stat = ((x1_E - x2_E) / (S**0.5)) * ((x1_n * x2_n / (x1_n + x2_n))**0.5)
    if verbose:
        print(stat)
    return bool(stat > 1.96)


#------------------------------------------------------------------------------------------

t_test(ctr1, ctr2, verbose=True)
# [OUT]: True => отвергаем H0

#------------------------------------------------------------------------------------------

t_test(last1, last2, verbose=True)

# [OUT]: False => принимаем H0

#------------------------------------------------------------------------------------------


def bootstrap(x1, x2, k_iterations=5000, n_samples=25000):
    x1_inds = range(x1.shape[0])
    x2_inds = range(x2.shape[0])

    t_res_sum = 0.0
    for _ in range(k_iterations):
        x1_sample = x1[np.random.choice(x1_inds, n_samples, replace=True)]
        x2_sample = x2[np.random.choice(x2_inds, n_samples, replace=True)]

        t_res_sum += t_test(x1_sample, x2_sample)

    return t_res_sum / k_iterations


#------------------------------------------------------------------------------------------

# H0: CRT на двух сплитах можно объяснить шумами
bootstrap(ctr1, ctr2)
# [OUT]: 1.0 => отвергаем H0

#------------------------------------------------------------------------------------------

# H0: last click pos на двух сплитах можно объяснить шумами
bootstrap(last1, last2)
# [OUT]: 0.023 => принимаем H0