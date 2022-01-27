# Implementa o algoritmo Nested Benders Decomposition para resolver o problema do exemplo 1, da página 270

from pyomo.environ import *
import sys

def pde(file, K, pred):
    # Estágio de cada cenário
    stages = [1]
    for k in range(1, K):
        stages.append(stages[pred[k]] + 1)
    H = stages[K - 1]                               # número de estágios
    print(stages)
    print(H)

    model = AbstractModel("pde")
    
    # Conjuntos
    model.P1 = Set()                                # cargas que levam um estágio para chegar
    model.P2 = Set()                                # cargas que levam dois estágios para chegar
    model.P = Set(initialize=model.P1 | model.P2)
    model.T = RangeSet(H)                           # estágios
    model.S = Set()                                 # cenários

    # Parâmetros
    model.p = Param(model.S)                    # probabilidade de cada cenário
    model.d = Param(model.S)                    # demanda de cada cenário
    model.f = Param(model.P)                    # custo unitário de cada carga
    model.q = Param(model.P)                    # volume de cada carga
    model.a = Param(model.T)                    # volume adquirido anteriormente que chega em cada estágio
    model.h = Param(model.T)                    # custo unitário de estoque em cada estágio
    model.sMin = Param()                        # estoque mínimo
    model.sMax = Param()                        # estoque máximo

    # Variáveis
    model.s = Var(model.S, domain=NonNegativeReals, bounds=(model.sMin, model.sMax))    # estoque ao final de cada cenário
    model.u = Var(model.S, domain=NonNegativeReals)     # volume adquirido que chega em cada cenário
    model.v = Var(model.P, model.S, domain=Binary)      # se a carga c foi adquirida no cenário s

    # Função objetivo
    def objetivo(model):
        return sum(model.p[s] * (sum(model.f[c]*model.q[c]*model.v[c, s] for c in model.P) +\
        model.h[stages[s]]*model.s[s]) for s in model.S)
    model.OBJ = Objective(rule=objetivo)

    # Restrições
    def balanco(model, s):
        if stages[s] > 1:       # caso geral
            return model.a[stages[s]] + model.s[pred[s]] + model.u[s] == model.d[s] + model.s[s]
        else:                   # primeiro estágio
            return model.a[stages[s]] == model.d[s] + model.s[s]
    model.balanco = Constraint(model.S, rule=balanco)

    def chegada(model, s):
        if stages[s] == 1:      # primeiro estágio
            return model.u[s] == 0
        elif stages[s] == 2:    # segundo estágio
            return model.u[s] == sum(model.q[c]*model.v[c, pred[s]] for c in model.P1)
        else:
            return model.u[s] == sum(model.q[c]*model.v[c, pred[s]] for c in model.P1) +\
                sum(model.q[c]*model.v[c, pred[pred[s]]] for c in model.P2)
    model.chegada = Constraint(model.S, rule=chegada)

    # Resolve o modelo e imprime o resultado
    opt = SolverFactory("glpk")
    instance = model.create_instance(file)
    opt.solve(instance)

    instance.pprint()

    print(f"\n\n***SOLUÇÃO ÓTIMA ENCONTRADA***\n\nz* = {value(instance.OBJ)}")
    for s in instance.S:
        print(f"\nCenário {s}:\ns = {value(instance.s[s])}")
        print(f"u = {value(instance.u[s])}")
        print(f"v = {[value(instance.v[c, s]) for c in instance.P]}")

if __name__ == "__main__":
    K = int(sys.argv[2])
    pred = []
    for k in range(K):
        pred.append(int(sys.argv[k + 3]))
    print(K)
    print(pred)
    pde(sys.argv[1], K, pred)
