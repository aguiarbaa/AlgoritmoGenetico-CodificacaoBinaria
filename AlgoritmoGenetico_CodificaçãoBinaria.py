#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:48:33 2019

@author: edwiges

"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

n = 5
l = 36
pc = 0.8
pm = 0.025
ng = 50
iteracao = 1

def inicializa (n):
    populacaoInicial = np.zeros([n,l])
    for i in range(n):
        for j in range(l):
            populacaoInicial[i][j] = 0
            if np.random.rand() > 0.5:
                populacaoInicial[i][j] = 1
    return populacaoInicial

def fitness (populacao):
    populacao = np.array(populacao)
    fit = np.zeros([populacao.shape[0]]) 
    for i in range(populacao.shape[0]):
        for j in range(populacao.shape[1]):
            fit[i] = (9
            + populacao[i][1]
            *populacao[i][4]   - populacao[i][22]*populacao[i][13] + populacao[i][23]*populacao[i][3]  - populacao[i][20]*populacao[i][9]
            + populacao[i][35]*populacao[i][14] - populacao[i][10]*populacao[i][25] + populacao[i][15]*populacao[i][16] + populacao[i][2]*populacao[i][32]
            + populacao[i][27]*populacao[i][18] + populacao[i][11]*populacao[i][33] - populacao[i][30]*populacao[i][31] - populacao[i][21]*populacao[i][24]
            + populacao[i][34]*populacao[i][26] - populacao[i][28]*populacao[i][6]  + populacao[i][7]*populacao[i][12]  - populacao[i][5]*populacao[i][8]
            + populacao[i][17]*populacao[i][19] - populacao[i][0]*populacao[i][29]  + populacao[i][22]*populacao[i][3]  + populacao[i][20]*populacao[i][14]
            + populacao[i][25]*populacao[i][15] + populacao[i][30]*populacao[i][11] + populacao[i][24]*populacao[i][18] + populacao[i][6]*populacao[i][7]
            + populacao[i][8]*populacao[i][17]  + populacao[i][0]*populacao[i][32]
            )
    
    return fit, np.mean(fit)

def selecao_roleta(populacao,n):
    
    fit, media             = fitness(populacao)
    probabilidade          = np.zeros([fit.shape[0]])
    probabilidadeAcumulada = np.zeros([fit.shape[0]])
    populacao_s            = np.zeros([n,l])
    

    for i in range(fit.shape[0]):
        probabilidade[i] = fit[i]/sum(fit)
    
    
    probabilidadeAcumulada = [sum(probabilidade[:i+1]) for i in range(len(probabilidade))]
    
    for i in range(len(probabilidadeAcumulada)):
        
        numeroAleatorio = np.random.random_sample()         
        
        for j in range(n):           
            if numeroAleatorio <= probabilidadeAcumulada[j]:
                populacao_s[i] = populacao[j]
                break
    
    return np.array(populacao_s)
    
def selecao_torneio(populacao,n):
    populacao_s = np.zeros([n,l])
    
    for i in range(n):            
        individuoA       = np.random.randint(n-1)
        individuoB       = np.random.randint(n-1)
        populacaoTorneio = ([populacao[individuoA],populacao[individuoB]])
        fitnessSelecionados, media = fitness(populacaoTorneio)
    
        if fitnessSelecionados[0] >= fitnessSelecionados[1]:
            populacao_s[i] = populacao[individuoA]
        else:
            populacao_s[i] = populacao[individuoB]
    return np.array(populacao_s)

def selecao (populacao, n, opcao):
    if not opcao:
        return selecao_roleta(populacao,n)
    else:
        return selecao_torneio(populacao,n)

def cruzamento_un(paiA, paiB, l):        
    filho1 = np.zeros([l])
    filho2 = np.zeros([l])
    mascara = inicializa(1)

    for i in range(mascara.shape[1]):
        if mascara[0,i] == 0:
            filho1[i] = paiB[i]
            filho2[i] = paiA[i]            
        else:
            filho1[i] = paiA[i]
            filho2[i] = paiB[i]
          
    return filho1, filho2

def cruzamento_pc(paiA, paiB, l):
    pontoCorte = np.random.randint(l)
    filho1 = np.zeros([l]) 
    filho2 = np.zeros([l]) 
    for i in range(l):
        if i < pontoCorte:
            filho1[i] = paiA[i]
            filho2[i] = paiB[i]
        else:
            filho1[i] = paiB[i]
            filho2[i] = paiA[i]
    
    return filho1, filho2

def cruzamento(populacao_s, pc, opcao):
    populacao_filha = np.zeros([n,l])   
    escolhas        = np.random.choice(n,n)
    filho1          = [()]
    filho2          = [()]
    
    for i in range(0,n,2):
        r = np.random.rand()
        if r <= pc:        
            if not opcao:       
                filho1, filho2 =  cruzamento_pc(populacao_s[escolhas[i]], populacao_s[escolhas[i+1]],l)
            else:
                filho1, filho2 =  cruzamento_un(populacao_s[escolhas[i]], populacao_s[escolhas[i+1]],l)
            
            for j in range(l):
                populacao_filha[i][j]   = filho1[j]
                populacao_filha[i+1][j] = filho2[j]            
        else:
            for j in range(l):      
                populacao_filha[i][j]   = populacao_s[escolhas[i]][j]                
                populacao_filha[i+1][j] = populacao_s[escolhas[i+1]][j]
    
    return np.array(populacao_filha)

def mutacao_bit_bit(populacao_f, pm):
    populacao_fm = populacao_f  
    for i in range(populacao_fm.shape[0]):
        for j in range(populacao_fm.shape[1]):
            r = np.random.rand()
            if r <= pm:
                if populacao_fm[i][j] == 0:
                    populacao_fm[i][j] = 1 
                else:
                    populacao_fm[i][j] = 0    
                    
    return np.array(populacao_fm)
 
def mutacao_bit_aleatoria(populacao_f, l):
    populacao_fm = populacao_f
    for i in range(populacao_fm.shape[0]):
        bit_aleatorio = np.random.randint(0,l)
        if populacao_fm[i][bit_aleatorio] == 0:
            populacao_fm[i][bit_aleatorio] = 1
        else:
            populacao_fm[i][bit_aleatorio] = 0
    
    return np.array(populacao_fm)

def mutacao(populacao_f, pm, l, opcao):
    if not opcao:
        return mutacao_bit_aleatoria(populacao_f,l)
    else: 
        return mutacao_bit_bit(populacao_f, pm)
    
def substituicao_elitismo(populacao_fm):
    fit, media   = fitness(populacao_fm)
    melhores_elementos = np.zeros([n])
    populacao_evoluida = np.zeros([n,l])
    
    for i in range(n):
        melhores_elementos[i] = fit.argmax()
        fit[fit.argmax()] = 0
        populacao_evoluida[i] = populacao_fm[int(melhores_elementos[i])]
    
    return populacao_evoluida

def substituicao(populacao_mutante, populacao_filha, opcao):
    if not opcao:
        return substituicao_elitismo(populacao_mutante)
    else: 
        return np.array(populacao_filha)
    
""" variáveis globais """
n = 30
l = 36
pc = 0.8
pm = 0.025
ng = 50
iteracao = 100

""" O parâmetro opção deve ser setado de acordo com o teste a ser executado  """

def main (opcao):
    ev = 0
    evolucao_fit    = np.zeros([ng])
    maior_fitness = np.zeros([iteracao])
    menor_fitness = np.zeros([iteracao])
    media_fitness = np.zeros([iteracao])
    sucesso = 0
    for i in range(iteracao):
        def AlgoritmoGenetico(n, l, pc, pm, ng, opcao):
            t = 0            
            populacao = inicializa(n)            
            fitness(populacao)
            while t < ng:
                
                """ roleta:  0
                    torneio: 1 """
                populacao_selecionada  = selecao(populacao, n, 1)
                
                nova_populacao = np.zeros([n*2, l])   
                """ uniforme:       1
                    ponto de corte: 0 """
                populacao_filha = cruzamento(populacao_selecionada, pc, 0)
                
                for i in range(n):
                    nova_populacao[i]   = populacao_filha[i]    
                    nova_populacao[n+i] = populacao_selecionada[i]
                
                """ aleatorio: 0
                    bit a bit: 1 """
                populacao_mutante = (mutacao(nova_populacao, opcao, l, 1))
        
                """ elitismo:     0
                    sem elitismo: 1"""
                populacao = np.zeros([n,l])
                populacao = substituicao(populacao_mutante, populacao_filha, opcao)
                                
                fit, media    = fitness(populacao) 
                
                evolucao_fit[t] = evolucao_fit[t] + fit[fit.argmax()]
                                
                t = t + 1
                
            return populacao, evolucao_fit
            
        p, evolucao_fitness = AlgoritmoGenetico(n,l,pc,pm,ng, opcao)
        
        ev = evolucao_fitness/iteracao
        f, m = fitness(p)
        
        maior_fitness[i] = np.max(f)
        menor_fitness[i] = np.min(f)
        media_fitness[i] = np.mean(f)
        if np.max(f) == 27:
            sucesso = sucesso + 1
        print('iteração, sucesso: ',i, sucesso, np.max(f), opcao)
    return ev, sucesso, maior_fitness, menor_fitness, media_fitness


evolucao_fitness_c = main(0)
evolucao_fitness_s = main(1)

plt.figure()

evolucao_fitness_c = np.insert(evolucao_fitness_c,0,0)
evolucao_fitness_s = np.insert(evolucao_fitness_s,0,0)

plt.title('Evolução do Fitness')
plt.xlabel('Número de Gerações')
plt.ylabel('Valor do Fitness')
plt.plot(evolucao_fitness_c,'-r', label='Com Elitismo')   
plt.plot(evolucao_fitness_s,'-b', label='Sem Elitismo')   

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))

plt.legend(by_label.values(),by_label.keys())

plt.yticks([int(i) for i in range(0,30,3)])
plt.xticks([int(i) for i in range(0,55,5)])
plt.savefig('TesteElitismo.png')

plt.plot()


























