# Resolve o problema determinístico equivalente do Lot Sizing Estocástico. Assume que:
# * só chegam cargas pré-adquiridas nos dois primeiros estágios
# * todas as cargas em P chegam em um estágio
# * todas as cargas em A podem ser canceladas ou adiadas com um estágio de antecedência. Se adiadas, chegam em um estágio.

from pyomo.environ import *
import sys, time, os

TIME_LIMIT = 3600                               # limite de tempo em segundos

def pde(file, H, g):
    K = int((g**H - 1) / (g - 1))               # número de cenários
    pred = [-1]                                 # predecessor de cada cenário
    stages = [1]                                # estágio de cada cenário
    for k in range(1, K):
        pred.append((k - 1) // g)
        stages.append(stages[pred[k]] + 1)
    
    start_time = time.time()

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
    model.s0 = Param()                          # estoque inicial antes do primeiro estágio

    # Variáveis
    model.s = Var(model.S, domain=NonNegativeReals, bounds=(model.sMin, model.sMax))    # estoque ao final de cada cenário
    model.u = Var(model.S, domain=NonNegativeReals)     # volume adquirido que chega em cada cenário
    model.v = Var(model.P, model.S, domain=NonNegativeReals, bounds=(0, 1))     # se a carga c é adquirida no cenário s
    model.w = Var(model.S, domain=NonNegativeReals)     # volume cancelado que chegaria em cada cenário (só se aplica ao segundo estágio)
    model.x = Var(model.A2, domain=NonNegativeReals, bounds=(0, 1)) # se a carga c chegaria no segundo estágio e é cancelada
    model.y = Var(model.S, domain=NonNegativeReals)     # volume adiado para cada cenário (só se aplica ao terceiro estágio)
    model.z = Var(model.A2, domain=NonNegativeReals, bounds=(0, 1)) # se a carga c chegaria no segundo estágio e é adiada para o terceiro

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
            return sum(model.q[c] for c in model.A1) + model.s0 == model.d[s] + model.s[s]
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
    print("Criando instância")
    opt = SolverFactory("cplex")
    instance = model.create_instance(file, report_timing=True)
    build_time = time.time()
    if build_time - start_time > TIME_LIMIT:
        print("Time limit alcançado antes de começar a resolver!")
        return
    #instance.pprint()
    print("Resolvendo")
    opt.options['timelimit'] = TIME_LIMIT
    opt.solve(instance, tee=True)
    solution_time = time.time()
    #instance.display()

    print(f"\n\n***FIM DA EXECUÇÃO***\n\nz* = {value(instance.OBJ)}")
    print(f"Tempo de execução: {solution_time - start_time}s - Construção: {build_time - start_time}s; Solução: {solution_time - build_time}s")
    f = open(f"saida/{os.path.basename(file)}.txt", 'w')
    f.write(f"***FIM DA EXECUÇÃO***\n\nz* = {value(instance.OBJ)}\n")
    f.write(f"Tempo de execução: {solution_time - start_time}s - Construção: {build_time - start_time}s; Solução: {solution_time - build_time}s\n")
    for s in instance.S:
        f.write(f"\nCenário {s}:\ns = {value(instance.s[s])}\n")
        if stages[s] > 1:
            f.write(f"u = {value(instance.u[s])}\n")
        if stages[s] < H:
            f.write(f"v = {[value(instance.v[c, s]) for c in instance.P]}\n")
        if stages[s] == 2:
            f.write(f"w = {value(instance.w[s])}\n")
        if stages[s] == 1:
            f.write(f"x = {[value(instance.x[c]) for c in instance.A2]}\n")
        if stages[s] == 3:
            f.write(f"y = {value(instance.y[s])}\n")
        if stages[s] == 1:
            f.write(f"z = {[value(instance.z[c]) for c in instance.A2]}\n")

if __name__ == "__main__":
    pde(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
