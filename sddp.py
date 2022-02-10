# Implementa o algoritmo SDDP para um problema de Lot Sizing Estocástico. Assume que:
# * só chegam cargas pré-adquiridas nos dois primeiros estágios
# * a carga encomendada chega em um estágio
# * a carga pré-adquirida pode ser cancelada ou adiada com um estágio de antecedência. Se adiada, chega em um estágio.

from curses import nonl
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import sys, time, os
from random import random

EPSILON = 1e-5          # tolerância para os testes de otimalidade
ZALPHA2 = 2.326         # valor de z alpha/2 para 98% de confiança
L = 0                   # limite inferior para a função recurso
Q = 1e6                 # penalidade das variáveis artificiais phi

def sddp(file, H, M):
    # Retorna o modelo para o estágio t
    def criaModelo(t):
        model = AbstractModel(f"estagio{t}")

        # Conjuntos
        model.P = Set()                             # cargas disponíveis para aquisição neste estágio
        model.A = Set()                             # cargas já adquiridas anteriormente que podem ser canceladas ou adiadas neste estágio
        model.PAnt = Set()                          # conjunto P do estágio anterior
        model.AAnt = Set()                          # conjunto A do estágio anterior
        model.A2Ant = Set()                         # conjunto A de dois estágios atrás
        model.C = Set()                             # conjunto de todas as cargas
        model.S = Set()                             # cenários deste estágio

        # Parâmetros
        model.p = Param(model.S)                    # probabilidade de cada cenário deste estágio
        model.d = Param(model.S)                    # demanda de cada cenário deste estágio
        model.ca = Param(model.C)                   # custo unitário de aquisição de cada carga
        model.cc = Param(model.C)                   # custo unitário de cancelamento de cada carga
        model.cp = Param(model.C)                   # custo unitário de adiamento1 de cada carga
        model.q = Param(model.C)                    # volume de cada carga
        model.h = Param()                           # custo unitário de estoque
        model.sMin = Param()                        # estoque mínimo
        model.sMax = Param()                        # estoque máximo
        model.s0 = Param()                          # estoque inicial antes do primeiro estágio

        # Variáveis
        model.s = Var(domain=NonNegativeReals)              # estoque ao final deste estágio
        if t > 0:
            model.u = Var(domain=NonNegativeReals)          # volume adquirido que chega neste estágio
            model.w = Var(domain=NonNegativeReals)          # volume cancelado que chegaria neste estágio
            if t > 1:
                model.y = Var(domain=NonNegativeReals)      # volume adiado para este estágio
            # Variáveis artificiais para garantir recurso completo
            model.phi1 = Var(domain=NonNegativeReals)
            model.phi2 = Var(domain=NonNegativeReals)
        if t < H - 1:           # até o penúltimo estágio
            model.v = Var(model.P, domain=NonNegativeReals)         # fração da carga c adquirida para chegar em t+1
            model.x = Var(model.A, domain=NonNegativeReals)         # fração da carga c que chegaria em t+1 e é cancelada
            if t < H - 2:
                model.z2 = Var(model.A, domain=NonNegativeReals)    # fração da carga c que chegaria em t+1 e é adiada
            if t > 0:
                model.z1 = Var(model.AAnt, domain=NonNegativeReals) # fração da carga c que chegaria em t e foi adiada para t+1
            model.theta = Var(bounds=(L, None))

        # Expressões (termos que variam a cada iteração)
        if t > 0:
            model.sAnt = Expression()                       # estoque do estágio anterior
            model.vAnt = Expression(model.PAnt)             # valores de v do estágio anterior
            model.xAnt = Expression(model.AAnt)             # valores de x do estágio anterior
            if t < H - 1:
                model.z2Ant = Expression(model.AAnt)        # valores de z2 do estágio anterior
            if t > 1:
                model.z1Ant = Expression(model.A2Ant)       # valores de z1 do estágio anterior
            model.dk = Expression()                         # demanda do cenário considerado

        # Função objetivo
        if t == 0:              # primeiro estágio
            def objetivo(model):
                return sum(model.ca[c]*model.q[c]*model.v[c] for c in model.P) +\
                    sum(model.cc[c]*model.q[c]*model.x[c] for c in model.A) +\
                    sum((model.cp[c] - model.cc[c])*model.q[c]*model.z2[c] for c in model.A) +\
                    model.h*model.s + model.theta
        elif t < H - 2:         # caso geral
            def objetivo(model):
                return sum(model.ca[c]*model.q[c]*model.v[c] for c in model.P) +\
                    sum(model.cc[c]*model.q[c]*model.x[c] for c in model.A) +\
                    sum((model.cp[c] - model.cc[c])*model.q[c]*model.z2[c] for c in model.A) +\
                    model.h*model.s + Q*model.phi1 + Q*model.phi2 + model.theta
        elif t == H - 2:        # penúltimo estágio
            def objetivo(model):
                return sum(model.ca[c]*model.q[c]*model.v[c] for c in model.P) +\
                    sum(model.cc[c]*model.q[c]*model.x[c] for c in model.A) +\
                    model.h*model.s + Q*model.phi1 + Q*model.phi2 + model.theta
        else:                   # último estágio
            def objetivo(model):
                return model.h*model.s + Q*model.phi1 + Q*model.phi2
        model.OBJ = Objective(rule=objetivo)

        # Restrições
        if t > 1:               # caso geral
            def balanco(model):
                return sum(model.q[c] for c in model.AAnt) + model.sAnt + model.u + model.y + model.phi1 ==\
                    model.dk + model.w + model.s + model.phi2
        elif t == 1:            # segundo estágio
            def balanco(model):
                return sum(model.q[c] for c in model.AAnt) + model.sAnt + model.u + model.phi1 ==\
                    model.dk + model.w + model.s + model.phi2
        else:                   # primeiro estágio
            def balanco(model):
                return sum(model.q[c] for c in model.AAnt) + model.s0 == model.d[model.S.at(1)] + model.s
        model.balanco = Constraint(rule=balanco)
        model.limiteSMin = Constraint(expr=model.s >= model.sMin)
        model.limiteSMax = Constraint(expr=model.s <= model.sMax)
        if t > 0:
            def aquisicao(model):
                return model.u == sum(model.q[c]*model.vAnt[c] for c in model.PAnt)
            model.aquisicao = Constraint(rule=aquisicao)
            def cancelamento(model):
                return model.w == sum(model.q[c]*model.xAnt[c] for c in model.AAnt)
            model.cancelamento = Constraint(rule=cancelamento)
            if t > 1:
                def adiamento1(model):
                    return model.y == sum(model.q[c]*model.z1Ant[c] for c in model.A2Ant)
                model.adiamento1 = Constraint(rule=adiamento1)
            if t < H - 1:
                def adiamento2(model, c):
                    return model.z1[c] == model.z2Ant[c]
                model.adiamento2 = Constraint(model.AAnt, rule=adiamento2)
        if t < H - 2:
            def cancelamentoAdiamento(model, c):
                return model.z2[c] <= model.x[c]
            model.cancelamentoAdiamento = Constraint(model.A, rule=cancelamentoAdiamento)
        if t < H - 1:
            def limiteV(model, c):
                return model.v[c] <= 1
            model.limiteV = Constraint(model.P, rule=limiteV)
            def limiteX(model, c):
                return model.x[c] <= 1
            model.limiteX = Constraint(model.A, rule=limiteX)
            if t < H - 2:
                def limiteZ2(model, c):
                    return model.z2[c] <= 1
                model.limiteZ2 = Constraint(model.A, rule=limiteZ2)

        if t < H - 1:
            model.cortesOtimalidade = ConstraintList()      # lista de cortes de otimalidade adicionados ao longo do algoritmo
        if t > 0:
            model.dual = Suffix(direction=Suffix.IMPORT)

        return model.create_instance(file, namespace=f"t{t}")

    # Cria os modelos
    opt = SolverFactory("glpk")
    models = [criaModelo(t) for t in range(H)]
    a = [sum(models[t].q[c] for c in models[t].AAnt) for t in range(H)]    # volume adquirido anteriormente que chega em cada estágio

    # Retorna uma lista com todos os cenários possíveis (sem amostragem)
    def geraTodosCenarios():
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
    
    # Retorna uma lista de M cenários amostrados aleatoriamente
    def geraAmostra():
        amostra = []
        m = 0
        while m < M:
            #print("Começando novo a")
            a = []
            for t in range(H):
                r = random()
                p = 0
                #print(f"Sorteou {r}")
                for s in models[t].S:
                    p += models[t].p[s]
                    if p >= r:
                        a.append(s)
                        break
                #print(f"a = {a}")
                
            # Verifica se essa amostra já não foi gerada (para não repetir)
            existe = False
            for a1 in amostra:
                existe = True
                for t in range(H):
                    if a1[t] != a[t]:
                        existe = False
                        break
                if existe:
                    #print("Já existe...")
                    break
            if not existe:
                amostra.append(a)
                m += 1
                #print(f"Adicionou. amostra = {amostra}")
        #input(f"Ta aí suas amostras juliette {amostra}")
        return amostra

    amostragem = M > 0      # se pediu 0 amostras, gera todos os cenários possíveis
    if not amostragem:
        amostra = geraTodosCenarios()
        M = len(amostra)
    
    # Soluções atuais de cada cenário amostrado
    sAtual = [[-1]*H for m in range(M)]
    vAtual = [[{} for t in range(H - 1)] for m in range(M)]
    xAtual = [[{} for t in range(H - 1)] for m in range(M)]
    z1Atual = [[{} for t in range(H - 1)] for m in range(M)]
    z2Atual = [[{} for t in range(H - 2)] for m in range(M)]

    # Soluções duais atuais do último estágio de cada amostra
    piAtual = [{} for m in range(M)]

    # Termos independentes dos cortes gerados ao longo do algoritmo, para cada subproblema
    eLists = [[] for t in range(H)]
    EsLists = [[] for t in range(H)]
    EvLists = [[] for t in range(H)]
    ExLists = [[] for t in range(H)]
    Ez2Lists = [[] for t in range(H)]
    Ez1Lists = [[] for t in range(H)]
    cortes_repetidos = 0
    
    # Resolve o problema para o cenário s do estágio t, usando a solução atual do estágio t-1 da amostra m.
    # Se s == None, é considerado o cenário do estágio t de m
    def resolveCenario(t, m, s=None):
        if s == None:
            s = amostra[m][t]

        #print(f"\nResolvendo problema ({t}, {s})")
        if t > 0:
            # Atualiza as expressões dos estágios anteriores
            models[t].sAnt.set_value(sAtual[m][t-1])
            for c in models[t].PAnt:
                models[t].vAnt[c].set_value(vAtual[m][t-1][c])
            for c in models[t].AAnt:
                models[t].xAnt[c].set_value(xAtual[m][t-1][c])
            if t > 1:
                for c in models[t].A2Ant:
                    models[t].z1Ant[c].set_value(z1Atual[m][t-1][c])
            if t < H - 1:
                for c in models[t].AAnt:
                    models[t].z2Ant[c].set_value(z2Atual[m][t-1][c])
            models[t].dk.set_value(models[t].d[s])
        #models[t].pprint()
        return opt.solve(models[t])
    
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
    def armazenaSolucao(m, t):
        sAtual[m][t] = value(models[t].s)
        if t < H - 1:
            vAtual[m][t] = {c: value(models[t].v[c]) for c in models[t].P}
            xAtual[m][t] = {c: value(models[t].x[c]) for c in models[t].A}
            if t < H - 2:
                z2Atual[m][t] = {c: value(models[t].z2[c]) for c in models[t].A}
            if t > 0:
                z1Atual[m][t] = {c: value(models[t].z1[c]) for c in models[t].AAnt}
        
        # Se último estágio, aproveita e coleta as duais
        if t == H - 1:
            piAtual[m] = obtemDuais(t)
        
        #imprimeSolucao(t)
    
    # Retorna a solução dual do estágio t
    def obtemDuais(t):
        duais = {"adiamento2": {}, "cancelamentoAdiamento": {}, "limiteV": {}, "limiteX": {}, "limiteZ2": {}, "cortesOtimalidade": []}
        for c in models[t].component_objects(Constraint, active=True):
            name = c.getname()
            if name in ["adiamento2", "cancelamentoAdiamento", "limiteV", "limiteX", "limiteZ2"]:
                for index in c:
                    duais[name][index] = models[t].dual[c[index]]
            elif name == "cortesOtimalidade":
                for index in c:
                    duais[name].append(models[t].dual[c[index]])
            else:
                for index in c:
                    duais[name] = models[t].dual[c[index]]
        return duais
    
    # Copia a solução do estágio t da amostra m1 para o correspondente da amostra m2
    def copiaSolucao(t, m1, m2):
        sAtual[m2][t] = sAtual[m1][t]
        if t < H - 1:       # até o penúltimo estágio
            vAtual[m2][t] = {c: vAtual[m1][t][c] for c in vAtual[m1][t]}
            xAtual[m2][t] = {c: xAtual[m1][t][c] for c in xAtual[m1][t]}
            if t < H - 2:
                z2Atual[m2][t] = {c: z2Atual[m1][t][c] for c in z2Atual[m1][t]}
            if t > 0:
                z1Atual[m2][t] = {c: z1Atual[m1][t][c] for c in z1Atual[m1][t]}

    def imprimeSolucao(t):
        f.write(f"Solução do estágio {t}: z* = {value(models[t].OBJ)}, s = {value(models[t].s)}")
        if t > 0:
            f.write(f", u = {value(models[t].u)}, w = {value(models[t].w)}")
            if t > 1:
                f.write(f", y = {value(models[t].y)}")
            f.write(f", phi1 = {value(models[t].phi1)}, phi2 = {value(models[t].phi2)}")
        if t < H - 1:       # até o penúltimo estágio
            f.write(f", theta = {value(models[t].theta)}\n")
            f.write(f"v = {[value(models[t].v[c]) for c in models[t].P]}\n")
            f.write(f"x = {[value(models[t].x[c]) for c in models[t].A]}\n")
            if t < H - 2:
                f.write(f"z2 = {[value(models[t].z2[c]) for c in models[t].A]}\n")
            if t > 0:
                f.write(f"z1 = {[value(models[t].z1[c]) for c in models[t].AAnt]}\n")
        else:
            f.write('\n')
    
    def equals(x, y):
        return abs(x - y) < EPSILON
    
    # Verifica se já foi adicionado um corte ao problema do estágio t com os mesmos coeficientes dados
    def corteExiste(t, Es, Ev, Ex, Ez2, Ez1, e):
        for i in range(len(eLists[t]) - 1, -1, -1):
            if equals(Es, EsLists[t][i]) and equals(e, eLists[t][i]):
                for c in Ev:
                    if not equals(Ev[c], EvLists[t][i][c]):
                        break
                else:
                    for c in Ex:
                        if not equals(Ex[c], ExLists[t][i][c]):
                            break
                    else:
                        for c in Ez2:
                            if not equals(Ez2[c], Ez2Lists[t][i][c]):
                                break
                        else:
                            for c in Ez1:
                                if not equals(Ez1[c], Ez1Lists[t][i][c]):
                                    break
                            else:
                                #print(f"Corte já existe pro {t}: e = {e}, Es = {Es}, Ev = {Ev}, Ex = {Ex}, Ez2 = {Ez2}, Ez1 = {Ez1}")
                                return True
        return False
    
    # Adiciona um corte de otimalidade de Benders agregado ao problema do estágio t, considerando a solução atual da amostra m
    # para este estágio
    def adicionaCorteBenders(m, t):
        #print(f"\nAdiciona corte de Benders para o estágio {t}, amostra {m} = {amostra[m]}")
        e = 0
        Es = 0
        Ev = {c: 0 for c in models[t].P}
        Ex = {c: 0 for c in models[t].A}
        Ez2 = {}
        Ez1 = {}
        if t < H - 2:
            Ez2 = {c: 0 for c in models[t].A}
        if t > 0:
            Ez1 = {c: 0 for c in models[t].AAnt}
        for s in models[t+1].S:
            if (t == H - 2) and (s == amostra[m][t+1]):         # este cenário já foi resolvido na fase forward
                duais = piAtual[m]
            else:
                resolveCenario(t + 1, m, s)
                duais = obtemDuais(t + 1)
            #print(f"duais = {duais}")

            sigma_e = sum(duais["cortesOtimalidade"][i] * eLists[t+1][i] for i in range(len(duais["cortesOtimalidade"])))
            #print(f"sigma_e = {sigma_e}")
            e += models[t+1].p[s] * (duais["balanco"]*(models[t+1].d[s] - a[t+1]) + duais["limiteSMin"]*models[t+1].sMin +
                duais["limiteSMax"]*models[t+1].sMax + sum(duais["limiteV"][d] for d in duais["limiteV"]) +
                sum(duais["limiteX"][d] for d in duais["limiteX"]) + sum(duais["limiteZ2"][d] for d in duais["limiteZ2"]) + sigma_e)
            Es += models[t+1].p[s] * duais["balanco"]
            for c in Ev:
                Ev[c] -= models[t+1].p[s] * models[t].q[c] * duais["aquisicao"]
            for c in Ex:
                Ex[c] -= models[t+1].p[s] * models[t].q[c] * duais["cancelamento"]
            for c in Ez2:
                Ez2[c] -= models[t+1].p[s] * duais["adiamento2"][c]
            for c in Ez1:
                Ez1[c] -= models[t+1].p[s] * models[t].q[c] * duais["adiamento1"]

        if corteExiste(t, Es, Ev, Ex, Ez2, Ez1, e):
            nonlocal cortes_repetidos
            cortes_repetidos += 1
        else:
            models[t].cortesOtimalidade.add(expr=Es*models[t].s +
                sum(Ev[c]*models[t].v[c] for c in Ev) +
                sum(Ex[c]*models[t].x[c] for c in Ex) +
                sum(Ez2[c]*models[t].z2[c] for c in Ez2) +
                sum(Ez1[c]*models[t].z1[c] for c in Ez1) + models[t].theta >= e)
            #print(f"Corte de Benders para o estágio {t}: theta >= {e} - {Es}s - (", end="")
            #for c in Ev:
            #    print(f"{Ev[c]}v{c} + ", end="")
            #print(") - (", end="")
            #for c in Ex:
            #    print(f"{Ex[c]}x{c} + ", end="")
            #print(") - (", end="")
            #for c in Ez2:
            #    print(f"{Ez2[c]}z2,{c} + ", end="")
            #print(") - (", end="")
            #for c in Ez1:
            #    print(f"{Ez1[c]}z1,{c} + ", end="")
            #print(")")
            eLists[t].append(e)
            EsLists[t].append(Es)
            EvLists[t].append(Ev)
            ExLists[t].append(Ex)
            Ez2Lists[t].append(Ez2)
            Ez1Lists[t].append(Ez1)
    
    LB = LBant = -1e9
    UB = 1e9

    # Critério de parada do algoritmo
    if amostragem:
        def criterioParada():       # critério estocástico
            return LB - LBant < EPSILON
    else:
        def criterioParada():       # critério exato
            return UB - LB < EPSILON

    iter = 0
    f = open(f"saida/{os.path.basename(file)}-M{M}.txt", 'w')
    start = time.time()
    while True:
        # Atualiza lower bound
        LBant = LB
        results = resolveCenario(0, 0, 0)
        if results.solver.termination_condition == TerminationCondition.infeasible:
            # Este trecho não será alcançado pois o problema é sempre viável
            print("Problema inviável!!!")
            return
        #models[0].pprint()
        #models[0].display()
        LB = value(models[0].OBJ)
        #print(f"\nLB = {LB}, UB = {UB}, LBant = {LBant}")
        if criterioParada():
            break                           # ótimo encontrado

        iter += 1
        print(f"LB = {LB}, UB = {UB}\n*** ITERAÇÃO {iter} ***")

        if amostragem:
            amostra = geraAmostra()

        print("\n* PASSO FORWARD *\n")
        armazenaSolucao(0, 0)
        media = 0
        somaprob = 0
        obj = [0]*M
        prob = [1]*M
        for m in range(M):
            #print(f"\nAmostra {m} - {amostra[m]}")
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
                    armazenaSolucao(m, t)
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
        desvio = (sum(prob[m] * (obj[m] - media)**2 for m in range(M)) / (M * somaprob))**0.5 if amostragem else 0
        UB = media + ZALPHA2 * desvio
        if criterioParada():
            break                           # ótimo encontrado

        print(f"\nLB = {LB}, UB = {UB}\n* PASSO BACKWARD *")
        # Demais estágios
        for t in range(H - 2, 0, -1):
            for m in range(M):
                if cenarioRepetido(m, t) == -1:
                    adicionaCorteBenders(m, t)
                    
        # Primeiro estágio
        adicionaCorteBenders(0, 0)
    
    f.write(f"***SOLUÇÃO ÓTIMA ENCONTRADA***\n\nTempo de execução: {time.time() - start}s\n")
    f.write(f"z* estocástico = {UB}\ngap estocástico = {UB} - {LB} = {UB - LB} ({(UB - LB)*100 / LB})%")
    f.write(f"Iterações: {iter}\nCortes gerados no estágio 0: {len(eLists[0])}; ")
    f.write(f"total de cortes: {sum(len(eList) for eList in eLists)}; cortes repetidos (não adicionados): {cortes_repetidos}")
    print(f"\n\n***SOLUÇÃO ÓTIMA ENCONTRADA***\n\nTempo de execução: {time.time() - start}s")
    print(f"z* estocástico = {UB}\ngap estocástico = {UB} - {LB} = {UB - LB} ({(UB - LB)*100 / LB})%")
    print(f"Iterações: {iter}\nCortes gerados no estágio 0: {len(eLists[0])}; ", end="")
    print(f"total de cortes: {sum(len(eList) for eList in eLists)}; cortes repetidos (não adicionados): {cortes_repetidos}")

    # Resolve o problema exato para obter uma solução viável
    amostra = geraTodosCenarios()
    M = len(amostra)
    sAtual = [[-1]*H for m in range(M)]
    vAtual = [[{} for t in range(H - 1)] for m in range(M)]
    xAtual = [[{} for t in range(H - 1)] for m in range(M)]
    z1Atual = [[{} for t in range(H - 1)] for m in range(M)]
    z2Atual = [[{} for t in range(H - 2)] for m in range(M)]
    piAtual = [{} for m in range(M)]
    UBexato = value(models[0].OBJ) - value(models[0].theta)
    for m in range(M):
        sAtual[m][0] = value(models[0].s)
        vAtual[m][0] = {c: value(models[0].v[c]) for c in models[0].P}
        xAtual[m][0] = {c: value(models[0].x[c]) for c in models[0].A}
        z2Atual[m][0] = {c: value(models[0].z2[c]) for c in models[0].A}
        for t in range(1, H):
            m1 = cenarioRepetido(m, t)
            if m1 == -1:        # cenário inédito
                results = resolveCenario(t, m)
                armazenaSolucao(m, t)
                p = 1
                for t1 in range(1, t + 1):
                    p *= models[t1].p[amostra[m][t1]]
                if t < H - 1:
                    UBexato += p * (value(models[t].OBJ) - value(models[t].theta))
                else:
                    UBexato += p * value(models[t].OBJ)
            else:               # cenário repetido
                copiaSolucao(t, m1, m)

    print(f"\nz* = {UBexato}\ngap = {UBexato} - {LB} = {UBexato - LB} ({(UBexato - LB)*100 / LB})%")
    print(f"\nEstágio 0:\ns = {value(models[0].s)}\nv = {[value(models[0].v[c]) for c in models[0].P]}")
    print(f"x = {[value(models[0].x[c]) for c in models[0].A]}")
    print(f"z2 = {[value(models[0].z2[c]) for c in models[0].A]}\ntheta = {value(models[0].theta)}")
    f.write(f"\nz* = {UBexato}\ngap = {UBexato} - {LB} = {UBexato - LB} ({(UBexato - LB)*100 / LB})%")
    f.write(f"\nEstágio 0:\ns = {value(models[0].s)}\nv = {[value(models[0].v[c]) for c in models[0].P]}\n")
    f.write(f"x = {[value(models[0].x[c]) for c in models[0].A]}\n")
    f.write(f"z2 = {[value(models[0].z2[c]) for c in models[0].A]}\ntheta = {value(models[0].theta)}\n")
    for t in range(1, H):
        f.write(f"\nEstágio {t}:\n")
        for m in range(M):
            if cenarioRepetido(m, t) == -1:
                f.write(f"\nAmostra {m}: {amostra[m]}")
                resolveCenario(t, m)
                imprimeSolucao(t)
    f.close()

# Modo de execução:
# python sddip.py <arquivo> <H> <M>
# arquivo: nome do arquivo de entrada
# H: número de estágios na instância
# M: número de amostras a serem realizadas por iteração
if __name__ == "__main__":
    sddp(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
