# Resolve o problema determinístico equivalente do Lot Sizing Estocástico. Assume que:
# * só chegam cargas pré-adquiridas nos dois primeiros estágios
# * todas as cargas em P chegam em um estágio
# * todas as cargas em A podem ser canceladas ou adiadas com um estágio de antecedência. Se adiadas, chegam em um estágio.

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
    model.P = Set()                             # cargas disponíveis para aquisição
    model.A1 = Set()                            # cargas já adquiridas que chegam no primeiro estágio
    model.A2 = Set()                            # cargas já adquiridas que chegam no segundo estágio
    model.C = Set()                             # conjunto de todas as cargas
    model.T = RangeSet(H)                       # estágios
    model.S = Set()                             # cenários

    # Parâmetros
    model.p = Param(model.S)                    # probabilidade de cada cenário
    model.d = Param(model.S)                    # demanda de cada cenário
    model.ca = Param(model.C)                   # custo unitário de aquisição de cada carga
    model.cc = Param(model.C)                   # custo unitário de cancelamento de cada carga
    model.cp = Param(model.C)                   # custo unitário de adiamento1 de cada carga
    model.q = Param(model.C)                    # volume de cada carga
    model.h = Param(model.T)                    # custo unitário de estoque em cada estágio
    model.sMin = Param()                        # estoque mínimo
    model.sMax = Param()                        # estoque máximo

    # Variáveis
    model.s = Var(model.S, domain=NonNegativeReals, bounds=(model.sMin, model.sMax))    # estoque ao final de cada cenário
    model.u = Var(model.S, domain=NonNegativeReals)     # volume adquirido que chega em cada cenário
    model.v = Var(model.P, model.S, domain=Binary)      # se a carga c é adquirida no cenário s
    model.w = Var(model.S, domain=NonNegativeReals)     # volume cancelado que chegaria em cada cenário (só se aplica ao segundo estágio)
    model.x = Var(model.A2, domain=Binary)              # se a carga c chegaria no segundo estágio e é cancelada
    model.y = Var(model.S, domain=NonNegativeReals)     # volume adiado para cada cenário (só se aplica ao terceiro estágio)
    model.z = Var(model.A2, domain=Binary)              # se a carga c chegaria no segundo estágio e é adiada para o terceiro

    # Função objetivo
    def objetivo(model):
        return sum(model.p[s] * (sum(model.ca[c]*model.q[c]*model.v[c, s] for c in model.P) +
                model.h[stages[s]]*model.s[s]) for s in model.S) +\
            sum(model.cc[c]*model.q[c]*model.x[c] for c in model.A2) +\
            sum((model.cp[c] - model.cc[c])*model.q[c]*model.z[c] for c in model.A2)
    model.OBJ = Objective(rule=objetivo)

    # Restrições
    def balanco(model, s):
        if stages[s] > 3:       # caso geral
            return model.s[pred[s]] + model.u[s] == model.d[s] + model.s[s]
        elif stages[s] == 3:    # terceiro estágio
            return model.s[pred[s]] + model.u[s] + model.y[s] == model.d[s] + model.s[s]
        elif stages[s] == 2:    # segundo estágio
            return sum(model.q[c] for c in model.A2) + model.s[pred[s]] + model.u[s] == model.d[s] + model.w[s] + model.s[s]
        else:                   # primeiro estágio
            return sum(model.q[c] for c in model.A1) == model.d[s] + model.s[s]
    model.balanco = Constraint(model.S, rule=balanco)

    def aquisicao(model, s):
        if stages[s] == 1:      # primeiro estágio
            return model.u[s] == 0
        else:
            return model.u[s] == sum(model.q[c]*model.v[c, pred[s]] for c in model.P)
    model.aquisicao = Constraint(model.S, rule=aquisicao)

    def cancelamento(model, s):
        if stages[s] == 2:      # segundo estágio
            return model.w[s] == sum(model.q[c]*model.x[c] for c in model.A2)
        else:
            return model.w[s] == 0
    model.cancelamento = Constraint(model.S, rule=cancelamento)

    def adiamento(model, s):
        if stages[s] == 3:      # terceiro estágio
            return model.y[s] == sum(model.q[c]*model.z[c] for c in model.A2)
        else:
            return model.y[s] == 0
    model.adiamento = Constraint(model.S, rule=adiamento)

    def cancelamentoAdiamento(model, c):
        return model.z[c] <= model.x[c]
    model.cancelamentoAdiamento = Constraint(model.A2, rule=cancelamentoAdiamento)

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
        print(f"w = {value(instance.w[s])}")
        if stages[s] == 1:
            print(f"x = {[value(instance.x[c]) for c in instance.A2]}")
        print(f"y = {value(instance.y[s])}")
        if stages[s] == 1:
            print(f"z = {[value(instance.z[c]) for c in instance.A2]}")

if __name__ == "__main__":
    K = int(sys.argv[2])
    pred = []
    for k in range(K):
        pred.append(int(sys.argv[k + 3]))
    print(K)
    print(pred)
    pde(sys.argv[1], K, pred)
