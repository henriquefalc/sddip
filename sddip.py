# Implementa o algoritmo SDDiP para um problema de Lot Sizing Estocástico. Assume que:
# * só chegam cargas pré-adquiridas nos dois primeiros estágios
# * todas as cargas em P chegam em um estágio
# * todas as cargas em A podem ser canceladas ou adiadas com um estágio de antecedência. Se adiadas, chegam em um estágio.

from pyomo.environ import *
from pyomo.opt import TerminationCondition
import sys
from random import random

EPSILON = 1e-5          # tolerância para os testes de otimalidade
ZALPHA2 = 2.326         # valor de z alpha/2 para 98% de confiança
L = 0                   # limite inferior para a função recurso
Q = 1000                # penalidade das variáveis artificiais phi

def sddip(file, H, M):
    # Retorna o modelo para o estágio t
    # Se LR == True, retorna a relaxação linear do modelo
    def criaModelo(t, LR=False):
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
            if LR:
                model.v = Var(model.P, domain=NonNegativeReals)
                model.x = Var(model.A, domain=NonNegativeReals)
                if t < H - 2:   # até o antepenúltimo estágio
                    model.z2 = Var(model.A, domain=NonNegativeReals)
                if t > 0:
                    model.z1 = Var(model.AAnt, domain=NonNegativeReals)
            else:
                model.v = Var(model.P, domain=Binary)           # se a carga c é adquirida e chega no próximo estágio
                model.x = Var(model.A, domain=Binary)           # se a carga c chegaria no próximo estágio e é cancelada
                if t < H - 2:
                    model.z2 = Var(model.A, domain=Binary)      # se a carga c chegaria no próximo estágio e é adiada
                if t > 0:
                    model.z1 = Var(model.AAnt, domain=Binary)   # se a carga c chegaria neste estágio e foi adiada para o próximo
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
                return sum(model.q[c] for c in model.AAnt) + model.sAnt + model.u + model.phi1 == model.dk + model.w + model.s + model.phi2
        else:                   # primeiro estágio
            def balanco(model):
                return sum(model.q[c] for c in model.AAnt) == model.d[model.S.at(1)] + model.s
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
        if LR:
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
        if (t == H - 1) or (LR and (t > 0)):
            model.dual = Suffix(direction=Suffix.IMPORT)

        return model.create_instance(file, namespace=f"t{t}")

    # Cria os modelos
    opt = SolverFactory("glpk")
    models = [criaModelo(t) for t in range(H)]
    modelsLR = [criaModelo(t, LR=True) for t in range(H - 1)]
    a = [sum(models[t].q[c] for c in models[t].AAnt) for t in range(H)]    # volume adquirido anteriormente que chega em cada estágio

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
    vAtual = [[{} for t in range(H - 1)] for m in range(M)]
    xAtual = [[{} for t in range(H - 1)] for m in range(M)]
    z1Atual = [[{} for t in range(H - 1)] for m in range(M)]
    z2Atual = [[{} for t in range(H - 2)] for m in range(M)]

    # Soluções duais atuais do último estágio de cada amostra
    piAtual = [{} for m in range(M)]

    # Termos independentes dos cortes gerados ao longo do algoritmo, para cada subproblema
    eLists = [[] for t in range(H)]
    
    # Resolve o problema para o cenário s do estágio t, usando a solução atual do estágio t-1 da amostra m.
    # Se s == None, é considerado o cenário do estágio t de m
    # Se LR == True, resolve a relaxação linear do problema
    def resolveCenario(t, m, s=None, LR=False):
        if s == None:
            s = amostra[m][t]

        if LR and (t < H - 1):
            model = modelsLR[t]
        else:
            model = models[t]

        print(f"\nResolvendo problema ({t}, {s})")
        if t > 0:
            # Atualiza as expressões dos estágios anteriores
            model.sAnt.set_value(sAtual[m][t-1])
            for c in model.PAnt:
                model.vAnt[c].set_value(vAtual[m][t-1][c])
            for c in model.AAnt:
                model.xAnt[c].set_value(xAtual[m][t-1][c])
            if t > 1:
                for c in model.A2Ant:
                    model.z1Ant[c].set_value(z1Atual[m][t-1][c])
            if t < H - 1:
                for c in model.AAnt:
                    model.z2Ant[c].set_value(z2Atual[m][t-1][c])
            model.dk.set_value(model.d[s])
        #model.pprint()
        return opt.solve(model)
    
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
        
        imprimeSolucao(t)
    
    # Retorna a solução dual do estágio t
    def obtemDuais(t):
        if t < H - 1:
            model = modelsLR[t]
        else:
            model = models[t]
        duais = {"adiamento2": {}, "cancelamentoAdiamento": {}, "limiteV": {}, "limiteX": {}, "limiteZ2": {}, "cortesOtimalidade": []}
        for c in model.component_objects(Constraint, active=True):
            name = c.getname()
            if name in ["adiamento2", "cancelamentoAdiamento", "limiteV", "limiteX", "limiteZ2"]:
                for index in c:
                    duais[name][index] = model.dual[c[index]]
            elif name == "cortesOtimalidade":
                for index in c:
                    duais[name].append(model.dual[c[index]])
            else:
                for index in c:
                    duais[name] = model.dual[c[index]]
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
        print(f"Solução do estágio {t}: z* = {value(models[t].OBJ)}, s = {value(models[t].s)}", end="")
        if t > 0:
            print(f", u = {value(models[t].u)}, w = {value(models[t].w)}", end="")
            if t > 1:
                print(f", y = {value(models[t].y)}", end="")
            print(f", phi1 = {value(models[t].phi1)}, phi2 = {value(models[t].phi2)}", end="")
        if t < H - 1:       # até o penúltimo estágio
            print(f", theta = {value(models[t].theta)}")
            print(f"v = {[value(models[t].v[c]) for c in models[t].P]}")
            print(f"x = {[value(models[t].x[c]) for c in models[t].A]}")
            if t < H - 2:
                print(f"z2 = {[value(models[t].z2[c]) for c in models[t].A]}")
            if t > 0:
                print(f"z1 = {[value(models[t].z1[c]) for c in models[t].AAnt]}")
        else:
            print()
    
    # Adiciona um corte de otimalidade de Benders agregado ao problema do estágio t, considerando a solução atual da amostra m
    # para este estágio
    def adicionaCorteBenders(m, t):
        print(f"\nAdiciona corte de Benders para o estágio {t}, amostra {m} = {amostra[m]}")
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
                resolveCenario(t + 1, m, s, LR=True)
                #modelsLR[t+1].pprint()
                #modelsLR[t+1].display()
                duais = obtemDuais(t + 1)
            print(f"duais = {duais}")

            sigma_e = 0
            for i in range(len(duais["cortesOtimalidade"])):
                sigma_e += duais["cortesOtimalidade"][i] * eLists[t+1][i]
            #print(f"sigma_e = {sigma_e}")
            e += models[t+1].p[s] * (duais["balanco"]*(models[t+1].d[s] - a[t+1]) + duais["limiteSMin"]*models[t+1].sMin +
                duais["limiteSMax"]*models[t+1].sMax + sum(duais["limiteV"][d] for d in duais["limiteV"]) +
                sum(duais["limiteX"][d] for d in duais["limiteX"]) + sum(duais["limiteZ2"][d] for d in duais["limiteZ2"]) + sigma_e)
            #print(f"e += {models[t+1].p[s]} * ({duais['balanco']}*({models[t+1].d[s]} - {models[t+1].a}) + {duais['limiteSMin']}*{models[t+1].sMin} + {duais['limiteSMax']}*{models[t+1].sMax} + {sum(duais["limiteV"][d] for d in duais['limiteV'])} + {sum(duais["limiteV2"][d] for d in duais['limiteV2'])} + {sigma_e}) = {models[t+1].p[s] * (duais['balanco']*(models[t+1].d[s] - models[t+1].a) + duais['limiteSMin']*models[t+1].sMin + duais['limiteSMax']*models[t+1].sMax + sum(duais["limiteV"][d] for d in duais['limiteV']) + sum(duais["limiteV2"][d] for d in duais['limiteV2']) + sigma_e)}")
            Es += models[t+1].p[s] * duais["balanco"]
            for c in Ev:
                Ev[c] -= models[t+1].p[s] * models[t].q[c] * duais["aquisicao"]
            for c in Ex:
                Ex[c] -= models[t+1].p[s] * models[t].q[c] * duais["cancelamento"]
            for c in Ez2:
                Ez2[c] -= models[t+1].p[s] * duais["adiamento2"][c]
            for c in Ez1:
                Ez1[c] -= models[t+1].p[s] * models[t].q[c] * duais["adiamento1"]

        models[t].cortesOtimalidade.add(expr=Es*models[t].s +
            sum(Ev[c]*models[t].v[c] for c in Ev) +
            sum(Ex[c]*models[t].x[c] for c in Ex) +
            sum(Ez2[c]*models[t].z2[c] for c in Ez2) +
            sum(Ez1[c]*models[t].z1[c] for c in Ez1) + models[t].theta >= e)
        modelsLR[t].cortesOtimalidade.add(expr=Es*modelsLR[t].s +
            sum(Ev[c]*modelsLR[t].v[c] for c in Ev) +
            sum(Ex[c]*modelsLR[t].x[c] for c in Ex) +
            sum(Ez2[c]*modelsLR[t].z2[c] for c in Ez2) +
            sum(Ez1[c]*modelsLR[t].z1[c] for c in Ez1) + modelsLR[t].theta >= e)
        print(f"Corte de Benders para o estágio {t}: theta >= {e} - {Es}s - (", end="")
        for c in Ev:
            print(f"{Ev[c]}v{c} + ", end="")
        print(") - (", end="")
        for c in Ex:
            print(f"{Ex[c]}x{c} + ", end="")
        print(") - (", end="")
        for c in Ez2:
            print(f"{Ez2[c]}z2,{c} + ", end="")
        print(") - (", end="")
        for c in Ez1:
            print(f"{Ez1[c]}z1,{c} + ", end="")
        print(")")
        eLists[t].append(e)
    
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
    
    print(f"\n\n***SOLUÇÃO ÓTIMA ENCONTRADA***\n\nz* = {UB}")
    print(f"\nEstágio 0:\ns = {value(models[0].s)}\nv = {[value(models[0].v[c]) for c in models[0].P]}")
    print(f"x = {[value(models[0].x[c]) for c in models[0].A]}")
    print(f"z2 = {[value(models[0].z2[c]) for c in models[0].A]}\ntheta = {value(models[0].theta)}")
    print(f"\nIterações: {iter}")
    input(f"gap = {UB} - {LB} = {UB - LB} ({(UB - LB)*100 / LB})%")
    for t in range(1, H):
        print(f"\nEstágio {t}:")
        for m in range(M):
            if cenarioRepetido(m, t) == -1:
                resolveCenario(t, m)
                imprimeSolucao(t)

# Modo de execução:
# python sddip.py <arquivo> <H> <M>
# arquivo: nome do arquivo de entrada
# H: número de estágios na instância
# M: número de amostras a serem realizadas por iteração
if __name__ == "__main__":
    sddip(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
