# Implementa o algoritmo SDDiP para um problema de Lot Sizing
# Versão com conjuntos P1 e P2 (e variáveis v1 e v2) e cortes L-shaped inteiros e de Benders fortalecidos (errados)

from pyomo.environ import *
from pyomo.opt import TerminationCondition
import sys
from random import random

EPSILON = 1e-5          # tolerância para os testes de otimalidade
ZALPHA2 = 2.326         # valor de z alpha/2 para 98% de confiança
L = 0                   # limite inferior para a função recurso
C = 1000                # penalidade das variáveis artificiais phi

def sddip(file, H, M):
    # Retorna o modelo para o estágio t
    # Se LR == True, retorna a relaxação linear do modelo
    # Se LRz == True, retorna a relaxação linear do modelo com restrições z == v
    # Se LagrSub == True, retorna o subproblema que define a relaxação Lagrangeana do problema com restrições z == v dualizadas
    def criaModelo(t, LR=False, LRz=False, LagrSub=False):
        model = AbstractModel(f"estagio{t}")

        # Conjuntos
        model.P1 = Set()                                # cargas que levam um estágio para chegar
        model.P2 = Set()                                # cargas que levam dois estágios para chegar
        model.P = Set(initialize=model.P1 | model.P2)
        model.S = Set()                                 # cenários deste estágio

        # Parâmetros
        model.p = Param(model.S)                    # probabilidade de cada cenário deste estágio
        model.d = Param(model.S)                    # demanda de cada cenário deste estágio
        model.f = Param(model.P)                    # custo unitário de cada carga
        model.q = Param(model.P)                    # volume de cada carga
        model.a = Param()                           # volume adquirido anteriormente que chega neste estágio
        model.h = Param()                           # custo unitário de estoque
        model.sMin = Param()                        # estoque mínimo
        model.sMax = Param()                        # estoque máximo

        # Variáveis
        model.s = Var(domain=NonNegativeReals)      # estoque ao final deste estágio
        if t > 0:
            model.u = Var(domain=NonNegativeReals)  # volume adquirido que chega neste estágio
            # Variáveis artificiais para garantir recurso completo
            model.phi1 = Var(domain=NonNegativeReals)
            model.phi2 = Var(domain=NonNegativeReals)
        if t < H - 1:       # até o penúltimo estágio
            if t > 0:
                if LR or LRz:
                    model.v1 = Var(model.P, domain=NonNegativeReals)
                else:
                    model.v1 = Var(model.P, domain=Binary)      # se a carga c foi adquirida e chega no próximo estágio
            else:
                if LR or LRz:
                    model.v1 = Var(model.P1, domain=NonNegativeReals)
                else:
                    model.v1 = Var(model.P1, domain=Binary)     # se a carga c foi adquirida e chega no próximo estágio
            if t < H - 2:   # até o antepenúltimo estágio
                if LR or LRz:
                    model.v2 = Var(model.P2, domain=NonNegativeReals)
                else:
                    model.v2 = Var(model.P2, domain=Binary)     # se a carga c foi adquirida e chega daqui a dois estágios
            model.theta = Var(bounds=(L, None))
        if (t > 0) and (LRz or LagrSub):
            if t > 1:
                model.zv1 = Var(model.P, domain=NonNegativeReals, bounds=(0, 1))
            else:
                model.zv1 = Var(model.P1, domain=NonNegativeReals, bounds=(0, 1))
            if t < H - 1:
                model.zv2 = Var(model.P2, domain=NonNegativeReals, bounds=(0, 1))

        # Expressões (termos que variam a cada iteração)
        if t > 0:
            model.sAnt = Expression()               # estoque do estágio anterior
            if t > 1:
                model.v1Ant = Expression(model.P)   # valores de v1 do estágio anterior
            else:       # segundo estágio
                model.v1Ant = Expression(model.P1)
            if t < H - 1:
                model.v2Ant = Expression(model.P2)  # valores de v2 do estágio anterior
            model.dk = Expression()                 # demanda do cenário considerado
            if LagrSub:
                # pi: vetor argumento da função Lagrangeana
                if t > 1:
                    model.piv1 = Expression(model.P)
                else:
                    model.piv1 = Expression(model.P1)
                if t < H - 1:
                    model.piv2 = Expression(model.P2)

        # Função objetivo
        if LagrSub:
            if t == 0:              # primeiro estágio
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) +\
                        sum(model.f[c]*model.q[c]*model.v2[c] for c in model.P2) + model.h*model.s + model.theta
            elif t == 1:            # segundo estágio
                if H == 3:          # t também é o penúltimo
                    def objetivo(model):
                        return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) + model.h*model.s +\
                            C*model.phi1 + C*model.phi2 + model.theta -\
                            sum(model.piv1[c]*model.zv1[c] for c in model.P1) - sum(model.piv2[c]*model.zv2[c] for c in model.P2)
                else:
                    def objetivo(model):
                        return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) +\
                            sum(model.f[c]*model.q[c]*model.v2[c] for c in model.P2) + model.h*model.s +\
                            C*model.phi1 + C*model.phi2 + model.theta -\
                            sum(model.piv1[c]*model.zv1[c] for c in model.P1) - sum(model.piv2[c]*model.zv2[c] for c in model.P2)
            elif t < H - 2:         # caso geral
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) +\
                        sum(model.f[c]*model.q[c]*model.v2[c] for c in model.P2) + model.h*model.s +\
                        C*model.phi1 + C*model.phi2 + model.theta - sum(model.piv1[c]*model.zv1[c] for c in model.P) -\
                        sum(model.piv2[c]*model.zv2[c] for c in model.P2)
            elif t == H - 2:        # penúltimo estágio
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) + model.h*model.s +\
                        C*model.phi1 + C*model.phi2 + model.theta - sum(model.piv1[c]*model.zv1[c] for c in model.P) -\
                        sum(model.piv2[c]*model.zv2[c] for c in model.P2)
            else:                   # último estágio
                def objetivo(model):
                    return model.h*model.s + C*model.phi1 + C*model.phi2 - sum(model.piv1[c]*model.zv1[c] for c in model.P)
        else:
            if t == 0:              # primeiro estágio
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) +\
                        sum(model.f[c]*model.q[c]*model.v2[c] for c in model.P2) + model.h*model.s + model.theta
            elif t < H - 2:         # caso geral
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) +\
                        sum(model.f[c]*model.q[c]*model.v2[c] for c in model.P2) + model.h*model.s +\
                        C*model.phi1 + C*model.phi2 + model.theta
            elif t == H - 2:        # penúltimo estágio
                def objetivo(model):
                    return sum(model.f[c]*model.q[c]*model.v1[c] for c in model.P1) + model.h*model.s +\
                        C*model.phi1 + C*model.phi2 + model.theta
            else:                   # último estágio
                def objetivo(model):
                    return model.h*model.s + C*model.phi1 + C*model.phi2
        model.OBJ = Objective(rule=objetivo)

        # Restrições
        if t > 0:               # caso geral
            def balanco(model):
                return model.a + model.sAnt + model.u + model.phi1 == model.dk + model.s + model.phi2
        else:                   # primeiro estágio
            def balanco(model):
                return model.a == model.d[model.S.at(1)] + model.s
        model.balanco = Constraint(rule=balanco)
        
        model.limiteSMin = Constraint(expr=model.s >= model.sMin)
        model.limiteSMax = Constraint(expr=model.s <= model.sMax)

        if t > 0:
            if t > 1:
                if LRz or LagrSub:
                    def chegada(model):
                        return model.u == sum(model.q[c]*model.zv1[c] for c in model.P)
                else:
                    def chegada(model):
                        return model.u == sum(model.q[c]*model.v1Ant[c] for c in model.P)
            else:               # segundo estágio
                if LRz or LagrSub:
                    def chegada(model):
                        return model.u == sum(model.q[c]*model.zv1[c] for c in model.P1)
                else:
                    def chegada(model):
                        return model.u == sum(model.q[c]*model.v1Ant[c] for c in model.P1)
            model.chegada = Constraint(rule=chegada)

            if t < H - 1:
                if LRz or LagrSub:
                    def carga2Estagios(model, c):
                        return model.v1[c] == model.zv2[c]
                else:
                    def carga2Estagios(model, c):
                        return model.v1[c] == model.v2Ant[c]
                model.carga2Estagios = Constraint(model.P2, rule=carga2Estagios)
            
            if (LRz or LagrSub) and (t > 0):
                def restricaoZv1(model, c):
                    return model.zv1[c] == model.v1Ant[c]
                if t > 1:
                    model.restricaoZv1 = Constraint(model.P, rule=restricaoZv1)
                else:
                    model.restricaoZv1 = Constraint(model.P1, rule=restricaoZv1)
                if t < H - 1:
                    def restricaoZv2(model, c):
                        return model.zv2[c] == model.v2Ant[c]
                    model.restricaoZv2 = Constraint(model.P2, rule=restricaoZv2)
        
        if (LR or LRz) and (t < H - 1):         # até o penúltimo estágio
            def limiteV1(model, c):
                return model.v1[c] <= 1
            if t > 0:
                model.limiteV1 = Constraint(model.P, rule=limiteV1)
            else:
                model.limiteV1 = Constraint(model.P1, rule=limiteV1)
            if t < H - 2:                       # até o antepenúltimo estágio
                def limiteV2(model, c):
                    return model.v2[c] <= 1
                model.limiteV2 = Constraint(model.P2, rule=limiteV2)

        if t < H - 1:
            model.cortesOtimalidade = ConstraintList()      # lista de cortes de otimalidade adicionados ao longo do algoritmo
        if (t == H - 1) or (t > 0 and (LR or LRz)):
            model.dual = Suffix(direction=Suffix.IMPORT)

        return model.create_instance(file, namespace=f"t{t}")

    # Cria os modelos
    opt = SolverFactory("glpk")
    models = [criaModelo(t) for t in range(H)]
    modelsLR = [criaModelo(t, LR=True) for t in range(H - 1)]
    #modelsLRz = [criaModelo(t, LRz=True) for t in range(H)]
    #modelsLagr = [criaModelo(t, LagrSub=True) for t in range(H)]

    # Retorna uma lista de M cenários amostrados aleatoriamente
    def geraAmostra():
        if M < 30:      # menos de 30 cenários: retorna todos os cenários possíveis (sem amostragem)
            def geraPerm(t):
                if t == 0:
                    return [[s for s in models[0].S]]
                perms = geraPerm(t - 1)
                res = []
                for perm in perms:
                    for s in models[t].S:
                        res.append(perm + [s])
                return res
            return geraPerm(H - 1)

        else:
            amostra = []
            m = 0
            while m < M:
                print("Começando novo a")
                a = []
                for t in range(H):
                    r = random()
                    p = 0
                    print(f"Sorteou {r}")
                    for s in models[t].S:
                        p += models[t].p[s]
                        if p >= r:
                            a.append(s)
                            break
                    print(f"a = {a}")
                    
                # Verifica se essa amostra já não foi gerada (para não repetir)
                existe = False
                for a1 in amostra:
                    existe = True
                    for t in range(H):
                        if a1[t] != a[t]:
                            existe = False
                            break
                    if existe:
                        print("Já existe...")
                        break
                if not existe:
                    amostra.append(a)
                    m += 1
                    print(f"Adicionou. amostra = {amostra}")
            return amostra

    amostra = geraAmostra()     # para amostragem por iteração, passar isso para dentro do loop
    M = len(amostra)
    
    # Soluções atuais de cada cenário amostrado
    sAtual = [[-1 for t in range(H)] for m in range(M)]
    v1Atual = [[{} for t in range(H)] for m in range(M)]
    v2Atual = [[{} for t in range(H)] for m in range(M)]

    # Soluções duais atuais do último estágio de cada amostra
    piAtual = [{} for m in range(M)]

    # Termos independentes dos cortes gerados ao longo do algoritmo, para cada subproblema
    eLists = [[] for t in range(H)]
    
    # Resolve o problema para o cenário s do estágio t, usando a solução atual do estágio t-1 da amostra m.
    # Se s == None, é considerado o cenário do estágio t de m
    # Se LR == True, resolve a relaxação linear do problema
    # Se LRz == True, resolve a relaxação linear do problema com restrições z == v
    # Se LagrSub == True, resolve o subproblema que define a relaxação Lagrangeana do problema com restrições z == v dualizadas.
    #   Neste caso, a função também recebe valor pi = (piv1, piv2), argumento da função Lagrangeana
    def resolveCenario(t, m, s=None, LR=False, LRz=False, LagrSub=False, pi=None):
        if s == None:
            s = amostra[m][t]
        
        #if LRz:
        #    model = modelsLRz[t]
        #elif LagrSub:
        #    model = modelsLagr[t]
        if LR and t < H - 1:
            model = modelsLR[t]
        else:
            model = models[t]

        print(f"\nResolvendo problema ({t}, {s})")
        if t > 0:
            # Atualiza as expressões dos estágios anteriores
            model.sAnt.set_value(sAtual[m][t-1])
            if t > 1:
                for c in model.P:
                    model.v1Ant[c].set_value(v1Atual[m][t-1][c])
            else:
                for c in model.P1:
                    model.v1Ant[c].set_value(v1Atual[m][t-1][c])
            if t < H - 1:
                for c in model.P2:
                    model.v2Ant[c].set_value(v2Atual[m][t-1][c])
            model.dk.set_value(model.d[s])
            if LagrSub:
                if t > 1:
                    for c in model.P:
                        model.piv1[c].set_value(pi[0][c])
                else:
                    for c in model.P1:
                        model.piv1[c].set_value(pi[0][c])
                if t < H - 1:
                    for c in model.P2:
                        model.piv2[c].set_value(pi[1][c])
        #if t == 1:
        #    model.pprint()
        #    input("Uai")
        return opt.solve(model)
        #if t == 1:
        #    model.display()
        #    input("Uai")
        #return res
    
    # Verifica se existe algum item na amostra anterior a m que é coincide com o item m até o estágio t
    # Se sim, retorna seu índice.
    def cenarioRepetido(m, t):
        if t == H - 1:      # no último estágio nunca repete
            return -1
        for m1 in range(m):
            existe = True
            for t1 in range(t + 1):
                if amostra[m1][t1] != amostra[m][t1]:
                    existe = False
                    break
            if existe:
                return m1
        return -1

    # Armazena a solução do estágio t da amostra m
    def armazenaSolucao(t, m):
        sAtual[m][t] = value(models[t].s)
        if t < H - 1:       # até o penúltimo estágio
            if t > 0:
                v1Atual[m][t] = {c: value(models[t].v1[c]) for c in models[t].P}
            else:
                v1Atual[m][t] = {c: value(models[t].v1[c]) for c in models[t].P1}
            if t < H - 2:   # até o antepenúltimo estágio
                v2Atual[m][t] = {c: value(models[t].v2[c]) for c in models[t].P2}
        
        # Se último estágio, aproveita e coleta as duais
        if t == H - 1:
            piAtual[m] = obtemDuais(t)
        
        imprimeSolucao(t)
    
    # Retorna a solução dual do estágio t
    def obtemDuais(t):
        if t < H - 1:
            model = modelsLR[t]
        else:
            model = models[t]
        duais = {"cortesOtimalidade": [], "carga2Estagios": {}, "limiteV1": {}, "limiteV2": {}}
        for c in model.component_objects(Constraint, active=True):
            name = c.getname()
            if name == "cortesOtimalidade":
                for index in c:
                    duais[name].append(model.dual[c[index]])
            elif name in ["carga2Estagios", "limiteV1", "limiteV2"]:
                for index in c:
                    duais[name][index] = model.dual[c[index]]
            else:
                for index in c:
                    duais[name] = model.dual[c[index]]
        return duais
    
    '''# Retorna o vetor dual do problema Lagrangeano associado às restrições z == s e z == v do estágio t
    def obtemDuaisLRz(t):
        duais = [{}, {}]         # (piv1, piv2)
        for c in modelsLRz[t].component_objects(Constraint, active=True):
            if c.getname() == "restricaoZv1":
                for index in c:
                    duais[0][index] = modelsLRz[t].dual[c[index]]
            elif c.getname() == "restricaoZv2":
                for index in c:
                    duais[1][index] = modelsLRz[t].dual[c[index]]
        return duais'''
    
    # Copia a solução do estágio t da amostra m1 para o correspondente da amostra m2
    def copiaSolucao(t, m1, m2):
        sAtual[m2][t] = sAtual[m1][t]
        if t < H - 1:       # até o penúltimo estágio
            v1Atual[m2][t] = {c: v1Atual[m1][t][c] for c in v1Atual[m1][t]}
            if t < H - 2:   # até o antepenúltimo estágio
                v2Atual[m2][t] = {c: v2Atual[m1][t][c] for c in v2Atual[m1][t]}

    def imprimeSolucao(t):
        print(f"Solução do estágio {t}: z* = {value(models[t].OBJ)}, s = {value(models[t].s)}", end="")
        if t > 0:
            print(f", u = {value(models[t].u)}, phi1 = {value(models[t].phi1)}, phi2 = {value(models[t].phi2)}", end="")
        if t < H - 1:       # até o penúltimo estágio
            print(f", theta = {value(models[t].theta)}")
            if t > 0:
                print(f"v1 = {[value(models[t].v1[c]) for c in models[t].P]}")
            else:
                print(f"v1 = {[value(models[t].v1[c]) for c in models[t].P1]}")
            if t < H - 2:   # até o antepenúltimo estágio
                print(f"v2 = {[value(models[t].v2[c]) for c in models[t].P2]}")
        else:
            print()
    
    # Adiciona um corte de otimalidade de Benders agregado ao problema do estágio t, considerando a solução atual da amostra m
    # para este estágio
    def adicionaCorteBenders(m, t):
        print(f"\nAdiciona corte de Benders para o estágio {t}, amostra {m} = {amostra[m]}")
        e = 0
        Es = 0
        if t > 0:
            Ev1 = {c: 0 for c in models[t].P}
        else:       # primeiro estágio
            Ev1 = {c: 0 for c in models[t].P1}
        if t < H - 2:
            Ev2 = {c: 0 for c in models[t].P2}
            for s in models[t+1].S:
                resolveCenario(t + 1, m, s, LR=True)
                #modelsLR[t+1].pprint()
                #modelsLR[t+1].display()
                duais = obtemDuais(t + 1)
                print(f"duais = {duais}")

                sigma_e = 0
                for i in range(len(duais["cortesOtimalidade"])):
                    #print(f"sigma_e += {duais['cortesOtimalidade'][i]} * {eLists[t+1][i]} = {duais['cortesOtimalidade'][i] * eLists[t+1][i]}")
                    sigma_e += duais["cortesOtimalidade"][i] * eLists[t+1][i]
                #print(f"sigma_e = {sigma_e}")
                e += models[t+1].p[s] * (duais["balanco"]*(models[t+1].d[s] - models[t+1].a) + duais["limiteSMin"]*models[t+1].sMin +
                    duais["limiteSMax"]*models[t+1].sMax + sum(duais["limiteV1"][d] for d in duais["limiteV1"]) +
                    sum(duais["limiteV2"][d] for d in duais["limiteV2"]) + sigma_e)
                #print(f"e += {models[t+1].p[s]} * ({duais['balanco']}*({models[t+1].d[s]} - {models[t+1].a}) + {duais['limiteSMin']}*{models[t+1].sMin} + {duais['limiteSMax']}*{models[t+1].sMax} + {sum(duais["limiteV1"][d] for d in duais['limiteV1'])} + {sum(duais["limiteV2"][d] for d in duais['limiteV2'])} + {sigma_e}) = {models[t+1].p[s] * (duais['balanco']*(models[t+1].d[s] - models[t+1].a) + duais['limiteSMin']*models[t+1].sMin + duais['limiteSMax']*models[t+1].sMax + sum(duais["limiteV1"][d] for d in duais['limiteV1']) + sum(duais["limiteV2"][d] for d in duais['limiteV2']) + sigma_e)}")
                Es += models[t+1].p[s] * duais["balanco"]
                if t > 0:
                    for c in models[t].P:
                        Ev1[c] -= models[t+1].p[s] * models[t+1].q[c] * duais["chegada"]
                else:
                    for c in models[t].P1:
                        Ev1[c] -= models[t+1].p[s] * models[t+1].q[c] * duais["chegada"]
                for c in models[t].P2:
                    #print(f"Ev2[{c}] -= {models[t+1].p[s]} * {duais['carga2Estagios'][c]} = {models[t+1].p[s] * duais['carga2Estagios'][c]}")
                    Ev2[c] -= models[t+1].p[s] * duais["carga2Estagios"][c]
                #print(f"Ev2 = {Ev2}")
            if t > 0:
                models[t].cortesOtimalidade.add(expr=Es*models[t].s + sum(Ev1[c]*models[t].v1[c] for c in models[t].P) +\
                    sum(Ev2[c]*models[t].v2[c] for c in models[t].P2) + models[t].theta >= e)
                modelsLR[t].cortesOtimalidade.add(expr=Es*modelsLR[t].s + sum(Ev1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P) +\
                    sum(Ev2[c]*modelsLR[t].v2[c] for c in modelsLR[t].P2) + modelsLR[t].theta >= e)
                #modelsLRz[t].cortesOtimalidade.add(expr=Es*modelsLRz[t].s + sum(Ev1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P) +\
                #    sum(Ev2[c]*modelsLRz[t].v2[c] for c in modelsLRz[t].P2) + modelsLRz[t].theta >= e)
                #modelsLagr[t].cortesOtimalidade.add(expr=Es*modelsLagr[t].s + sum(Ev1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P) +\
                #    sum(Ev2[c]*modelsLagr[t].v2[c] for c in modelsLagr[t].P2) + modelsLagr[t].theta >= e)
            else:
                models[t].cortesOtimalidade.add(expr=Es*models[t].s + sum(Ev1[c]*models[t].v1[c] for c in models[t].P1) +\
                    sum(Ev2[c]*models[t].v2[c] for c in models[t].P2) + models[t].theta >= e)
                modelsLR[t].cortesOtimalidade.add(expr=Es*modelsLR[t].s + sum(Ev1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P1) +\
                    sum(Ev2[c]*modelsLR[t].v2[c] for c in modelsLR[t].P2) + modelsLR[t].theta >= e)
                #modelsLRz[t].cortesOtimalidade.add(expr=Es*modelsLRz[t].s + sum(Ev1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P1) +\
                #    sum(Ev2[c]*modelsLRz[t].v2[c] for c in modelsLRz[t].P2) + modelsLRz[t].theta >= e)
                #modelsLagr[t].cortesOtimalidade.add(expr=Es*modelsLagr[t].s + sum(Ev1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P1) +\
                #    sum(Ev2[c]*modelsLagr[t].v2[c] for c in modelsLagr[t].P2) + modelsLagr[t].theta >= e)
            print(f"Corte de Benders para o estágio {t}: theta >= {e} - {Es}s - (", end="")
            for c in (models[t].P if t > 0 else models[t].P1):
                print(f"{Ev1[c]}v1,{c} + ", end="")
            print(") - (", end="")
            for c in models[t].P2:
                print(f"{Ev2[c]}v2,{c} + ", end="")
            print(")")
        else:
            for s in models[t+1].S:
                if s == amostra[m][t+1]:     # este cenário já foi resolvido na fase forward
                    duais = piAtual[m]
                else:
                    resolveCenario(t + 1, m, s, LR=True)
                    #models[t+1].pprint()
                    #models[t+1].display()
                    duais = obtemDuais(t + 1)
                print(f"duais = {duais}")
                
                e += models[t+1].p[s] * (duais["balanco"]*(models[t+1].d[s] - models[t+1].a) + duais["limiteSMin"]*models[t+1].sMin +
                    duais["limiteSMax"]*models[t+1].sMax + sum(duais["limiteV1"][d] for d in duais["limiteV1"]))
                Es += models[t+1].p[s] * duais["balanco"]
                for c in models[t+1].P:
                    Ev1[c] -= models[t+1].p[s] * models[t+1].q[c] * duais["chegada"]
            models[t].cortesOtimalidade.add(expr=Es*models[t].s + sum(Ev1[c]*models[t].v1[c] for c in models[t].P) +
                models[t].theta >= e)
            modelsLR[t].cortesOtimalidade.add(expr=Es*modelsLR[t].s + sum(Ev1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P) +
                modelsLR[t].theta >= e)
            #modelsLRz[t].cortesOtimalidade.add(expr=Es*modelsLRz[t].s + sum(Ev1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P) +
            #    modelsLRz[t].theta >= e)
            #modelsLagr[t].cortesOtimalidade.add(expr=Es*modelsLagr[t].s + sum(Ev1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P) +
            #    modelsLagr[t].theta >= e)
            print(f"Corte de Benders para o estágio {t}: theta >= {e} - {Es}s - (", end="")
            for c in models[t].P:
                print(f"{Ev1[c]}v1,{c} + ", end="")
            print(")")
        eLists[t].append(e)
    
    '''# Adiciona um corte de otimalidade L-shaped inteiro agregado ao problema do estágio t, considerando a solução atual da
    # amostra m para este estágio
    def adicionaCorteLShapedInteiro(m, t):
        print(f"\nAdiciona corte L-shaped inteiro para o estágio {t}, amostra {m} = {amostra[m]}")
        obj = 0
        for s in models[t+1].S:
            resolveCenario(t + 1, m, s)
            #models[t+1].pprint()
            #models[t+1].display()
            obj += models[t+1].p[s]*value(models[t+1].OBJ)
            print(f"obj = {obj}")
        print(f"v1 = {v1Atual[m][t]}, v2 = {v2Atual[m][t]}")
        if t > 0:
            print(f"apendando no e: {(obj - L)*(sum(v1Atual[m][t][c] for c in models[t].P) + sum(v2Atual[m][t][c] for c in models[t].P2)) - obj}")
            eLists[t].append((obj - L)*(sum(v1Atual[m][t][c] for c in models[t].P) + sum(v2Atual[m][t][c] for c in models[t].P2)) - obj)
            models[t].cortesOtimalidade.add(expr=(obj - L)*
                (sum((v1Atual[m][t][c] - 1)*models[t].v1[c] + (models[t].v1[c] - 1)*v1Atual[m][t][c] for c in models[t].P) +
                 sum((v2Atual[m][t][c] - 1)*models[t].v2[c] + (models[t].v2[c] - 1)*v2Atual[m][t][c] for c in models[t].P2)) -
                models[t].theta <= -obj)
            modelsLR[t].cortesOtimalidade.add(expr=(obj - L)*
                (sum((v1Atual[m][t][c] - 1)*modelsLR[t].v1[c] + (modelsLR[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLR[t].P) +
                 sum((v2Atual[m][t][c] - 1)*modelsLR[t].v2[c] + (modelsLR[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLR[t].P2)) -
                modelsLR[t].theta <= -obj)
            #modelsLRz[t].cortesOtimalidade.add(expr=modelsLRz[t].theta >= (obj - L)*
            #    (sum((v1Atual[m][t][c] - 1)*modelsLRz[t].v1[c] + (modelsLRz[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLRz[t].P) +
            #     sum((v2Atual[m][t][c] - 1)*modelsLRz[t].v2[c] + (modelsLRz[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLRz[t].P2)) + obj)
            #modelsLagr[t].cortesOtimalidade.add(expr=modelsLagr[t].theta >= (obj - L)*
            #    (sum((v1Atual[m][t][c] - 1)*modelsLagr[t].v1[c] + (modelsLagr[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLagr[t].P) +
            #     sum((v2Atual[m][t][c] - 1)*modelsLagr[t].v2[c] + (modelsLagr[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLagr[t].P2)) +
            #    obj)
        else:
            print(f"apendando no e: {(obj - L)*(sum(v1Atual[m][t][c] for c in models[t].P1) + sum(v2Atual[m][t][c] for c in models[t].P2)) - obj}")
            eLists[t].append((obj - L)*(sum(v1Atual[m][t][c] for c in models[t].P1) + sum(v2Atual[m][t][c] for c in models[t].P2)) - obj)
            models[t].cortesOtimalidade.add(expr=(obj - L)*
                (sum((v1Atual[m][t][c] - 1)*models[t].v1[c] + (models[t].v1[c] - 1)*v1Atual[m][t][c] for c in models[t].P1) +
                 sum((v2Atual[m][t][c] - 1)*models[t].v2[c] + (models[t].v2[c] - 1)*v2Atual[m][t][c] for c in models[t].P2)) -
                models[t].theta <= -obj)
            modelsLR[t].cortesOtimalidade.add(expr=(obj - L)*
                (sum((v1Atual[m][t][c] - 1)*modelsLR[t].v1[c] + (modelsLR[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLR[t].P1) +
                 sum((v2Atual[m][t][c] - 1)*modelsLR[t].v2[c] + (modelsLR[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLR[t].P2)) -
                modelsLR[t].theta <= -obj)
            #modelsLRz[t].cortesOtimalidade.add(expr=modelsLRz[t].theta >= (obj - L)*
            #    (sum((v1Atual[m][t][c] - 1)*modelsLRz[t].v1[c] + (modelsLRz[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLRz[t].P1) +
            #     sum((v2Atual[m][t][c] - 1)*modelsLRz[t].v2[c] + (modelsLRz[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLRz[t].P2)) + obj)
            #modelsLagr[t].cortesOtimalidade.add(expr=modelsLagr[t].theta >= (obj - L)*
            #    (sum((v1Atual[m][t][c] - 1)*modelsLagr[t].v1[c] + (modelsLagr[t].v1[c] - 1)*v1Atual[m][t][c] for c in modelsLagr[t].P1) +
            #     sum((v2Atual[m][t][c] - 1)*modelsLagr[t].v2[c] + (modelsLagr[t].v2[c] - 1)*v2Atual[m][t][c] for c in modelsLagr[t].P2)) +
            #    obj)
        print(f"Corte L-Shaped inteiro para o estágio {t}: theta >= ({obj} - {L})*((", end="")
        for c in models[t].P if t > 0 else models[t].P1:
            print(f"({v1Atual[m][t][c]} - 1)*v1,{c} + (v1,{c} - 1)*{v1Atual[m][t][c]} + ", end="")
        print(") + (", end="")
        for c in models[t].P2:
            print(f"({v2Atual[m][t][c]} - 1)*v2,{c} + (v2,{c} - 1)*{v2Atual[m][t][c]} + ", end="")
        print(f")) + {obj}")'''

    '''# Adiciona um corte de otimalidade de Benders fortalecido agregado ao problema do estágio t, considerando a solução atual da
    # amostra m para este estágio
    def adicionaCorteBendersFortalecido(m, t):
        print(f"\nAdiciona corte de Benders fortalecido para o estágio {t}, amostra {m} = {amostra[m]}")
        if t > 0:
            piv1 = {c: 0 for c in models[t].P}
        else:
            piv1 = {c: 0 for c in models[t].P1}
        if t < H - 2:
            piv2 = {c: 0 for c in models[t].P2}
        lagr = 0
        for s in models[t+1].S:
            # Resolve a RL com a restrição z == v
            resolveCenario(t + 1, m, s, LRz=True)
            piv1Atual, piv2Atual = obtemDuaisLRz(t + 1)
            print(f"piv1 = {piv1Atual}")
            if t > 0:
                for c in models[t].P:
                    piv1[c] += models[t+1].p[s]*piv1Atual[c]
            else:
                for c in models[t].P1:
                    piv1[c] += models[t+1].p[s]*piv1Atual[c]
            if t < H - 2:
                print(f"piv2 = {piv2Atual}")
                for c in models[t].P2:
                    piv2[c] += models[t+1].p[s]*piv2Atual[c]
            
            # Resolve o subproblema Lagrangeano usando o pi obtido
            resolveCenario(t + 1, m, s, LagrSub=True, pi=(piv1Atual, piv2Atual))
            #models[t+1].display()
            print(f"lagr += {value(modelsLagr[t+1].OBJ)}")
            lagr += models[t+1].p[s]*value(modelsLagr[t+1].OBJ)
        if t < H - 2:
            print(f"lagr = {lagr}, piv1 = {piv1}, piv2 = {piv2}")
        else:
            print(f"lagr = {lagr}, piv1 = {piv1}")
        if t > 0:
            if t < H - 2:
                models[t].cortesOtimalidade.add(expr=models[t].theta >= lagr +
                    sum(piv1[c]*models[t].v1[c] for c in models[t].P) + sum(piv2[c]*models[t].v2[c] for c in models[t].P2))
                modelsLR[t].cortesOtimalidade.add(expr=modelsLR[t].theta >= lagr +
                    sum(piv1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P) + sum(piv2[c]*modelsLR[t].v2[c] for c in modelsLR[t].P2))
                modelsLRz[t].cortesOtimalidade.add(expr=modelsLRz[t].theta >= lagr +
                    sum(piv1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P) + sum(piv2[c]*modelsLRz[t].v2[c] for c in modelsLRz[t].P2))
                modelsLagr[t].cortesOtimalidade.add(expr=modelsLagr[t].theta >= lagr +
                    sum(piv1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P) + sum(piv2[c]*modelsLagr[t].v2[c] for c in modelsLagr[t].P2))
            else:
                models[t].cortesOtimalidade.add(expr=models[t].theta >= lagr +
                    sum(piv1[c]*models[t].v1[c] for c in models[t].P))
                modelsLR[t].cortesOtimalidade.add(expr=modelsLR[t].theta >= lagr +
                    sum(piv1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P))
                modelsLRz[t].cortesOtimalidade.add(expr=modelsLRz[t].theta >= lagr +
                    sum(piv1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P))
                modelsLagr[t].cortesOtimalidade.add(expr=modelsLagr[t].theta >= lagr +
                    sum(piv1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P))
        else:
            models[t].cortesOtimalidade.add(expr=models[t].theta >= lagr +
                sum(piv1[c]*models[t].v1[c] for c in models[t].P1) + sum(piv2[c]*models[t].v2[c] for c in models[t].P2))
            modelsLR[t].cortesOtimalidade.add(expr=modelsLR[t].theta >= lagr +
                sum(piv1[c]*modelsLR[t].v1[c] for c in modelsLR[t].P1) + sum(piv2[c]*modelsLR[t].v2[c] for c in modelsLR[t].P2))
            modelsLRz[t].cortesOtimalidade.add(expr=modelsLRz[t].theta >= lagr +
                sum(piv1[c]*modelsLRz[t].v1[c] for c in modelsLRz[t].P1) + sum(piv2[c]*modelsLRz[t].v2[c] for c in modelsLRz[t].P2))
            modelsLagr[t].cortesOtimalidade.add(expr=modelsLagr[t].theta >= lagr +
                sum(piv1[c]*modelsLagr[t].v1[c] for c in modelsLagr[t].P1) + sum(piv2[c]*modelsLagr[t].v2[c] for c in modelsLagr[t].P2))
        print(f"Corte de Benders fortalecido para o estágio {t}: theta >= {lagr} + (", end="")
        for c in models[t].P if t > 0 else models[t].P1:
            print(f"{piv1[c]}v1,{c} + ", end="")
        print(") + (", end="")
        if t < H - 2:
            for c in models[t].P2:
                print(f"{piv2[c]}v2,{c} + ", end="")
        print(")")'''
    
    LB = LBant = -1e9
    UB = 1e9
    iter = 0
    while True:
        # Atualiza lower bound
        LBant = LB
        resolveCenario(0, 0, 0)
        #models[0].display()
        LB = value(models[0].OBJ)
        if (LB - LBant < EPSILON) or (UB - LB < EPSILON):  # critério para B ou B+I
        #if UB - LB < EPSILON:               # critério para I
            break                           # ótimo encontrado

        iter += 1
        input(f"\n*** ITERAÇÃO {iter} - LB = {LB}, UB = {UB}, LBant = {LBant}***")

        # Amostragem - descomentar para gerar uma amostra por iteração
        #amostra = geraAmostra()

        print("\n* PASSO FORWARD *\n")
        armazenaSolucao(0, 0)
        media = 0
        somaprob = 0
        obj = [0 for m in range(M)]
        prob = [1 for m in range(M)]
        for m in range(M):
            print(f"\nAmostra {m} - {amostra[m]}")
            if m > 0:
                copiaSolucao(0, 0, m)
            obj[m] = value(models[0].OBJ) - value(models[0].theta)
            for t in range(1, H):
                m1 = cenarioRepetido(m, t)
                if m1 == -1:        # cenário inédito
                    results = resolveCenario(t, m)
                    #models[t].display()
                    if results.solver.termination_condition == TerminationCondition.infeasible:
                        # Este trecho não será alcançado pois o problema é sempre viável
                        print("Problema inviável!!!")
                        return
                    armazenaSolucao(t, m)
                else:               # cenário repetido
                    copiaSolucao(t, m1, m)
                obj[m] += value(models[t].OBJ)
                if t < H - 1:
                    obj[m] -= value(models[t].theta)
                prob[m] *= models[t].p[amostra[m][t]]
            media += prob[m] * obj[m]
            somaprob += prob[m]
        
        # Atualiza upper bound
        media /= somaprob
        if media < UB:
            UB = media
        # Descomentar este trecho para usar upper bound estatístico
        #somavar = 0
        #for m in range(M):
        #    somavar += prob[m] * (obj[m] - media)**2
        #UB = media + ZALPHA2 * (somavar / (M * somaprob))**0.5
        input(f"\nLB = {LB}, UB = {UB}")
        #if (LB - LBant < EPSILON) or (UB - LB < EPSILON):  # critério para B ou B+I
        if UB - LB < EPSILON:               # critério para I
            break                           # ótimo encontrado

        print("\n* PASSO BACKWARD *")
        # Último estágio
        for m in range(M):
            if cenarioRepetido(m, H - 2) == -1:
                adicionaCorteBenders(m, H - 2)
                #adicionaCorteBendersFortalecido(m, H - 2)

        # Demais estágios
        for t in range(H - 3, 0, -1):
            for m in range(M):
                if cenarioRepetido(m, t) == -1:
                    adicionaCorteBenders(m, t)
                    #adicionaCorteLShapedInteiro(m, t)
                    #adicionaCorteBendersFortalecido(m, t)
                    
        # Primeiro estágio
        adicionaCorteBenders(0, 0)
        #adicionaCorteLShapedInteiro(0, 0)
        #adicionaCorteBendersFortalecido(0, 0)
    
    input(f"\n\n***SOLUÇÃO ÓTIMA ENCONTRADA***\n\nz* = {UB}")
    print(f"Estágio 0:\ns = {value(models[0].s)}\nv1 = {[value(models[0].v1[c]) for c in models[0].P1]}")
    print(f"v2 = {[value(models[0].v2[c]) for c in models[0].P2]}\ntheta = {value(models[0].theta)}")
    for m in range(M):
        print(f"\nAmostra {m}: {amostra[m]}")
        for t in range(1, H):
            resolveCenario(t, m)
            imprimeSolucao(t)
    print(f"\nIterações: {iter}")
    print(f"gap = {UB} - {LB} = {UB - LB} ({(UB - LB)*100 / LB})%")

# Modo de execução:
# python sddip.py <arquivo> <H> <M>
# arquivo: nome do arquivo de entrada
# H: número de estágios na instância
# M: número de amostras a serem realizadas por iteração
if __name__ == "__main__":
    sddip(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
