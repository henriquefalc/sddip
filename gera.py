# Gera uma instância para o problema, nos formatos do SDDiP e do PDE

import sys
from numpy import random

s0 = 20                         # estoque inicial
sMin = 0                        # limite mínimo do estoque
sMax = 80                       # limite máximo do estoque
h = 1                           # custo de estoque em cada estágio (fixo)
qMin = 10                       # valor mínimo de cada carga
qMax = 50                       # valor máximo de cada carga
caMin = 150                     # custo de aquisição mínimo de cada carga
caMax = 250                     # custo de aquisição máximo de cada carga
ccMin = 30                      # custo de cancelamento mínimo de cada carga
ccMax = 50                      # custo de cancelamento máximo de cada carga
cpMin = 5                       # custo de adiamento mínimo de cada carga
cpMax = 12                      # custo de adiamento máximo de cada carga
dMin = 10                       # demanda mínima de cada cenário
dMax = 50                       # demanda máxima de cada cenário

# Retorna um aleatório em [a, b)
def uniforme(a, b):
    return (b-a)*random.random() + a

def gera(id, H, g, A, P):
    def escreveSet(f, nome, valores, espacos=0):
        f.write(f"{' '*espacos}set {nome} :=")
        for i in valores:
            f.write(f" {i}")
        f.write(';\n')

    def escreveParam(f, nome, valor, espacos=0):
        f.write(f"{' '*espacos}param {nome} := {valor};\n")

    def escreveParamSet(f, nome, valores, espacos=0):
        f.write(f"{' '*espacos}param {nome} :=")
        for i in range(len(valores)):
            f.write(f"\n{' '*espacos}    {i+1} %.2f" % valores[i])
        f.write(';\n')

    C = A + P
    viavel = False      # viabilidade no primeiro estágio
    while not viavel:
        # Sorteia as cargas em A entre os dois primeiros estágios
        A1 = []
        A2 = []
        for c in range(1, A + 1):
            if random.random() < 0.5:
                A1.append(c)
            else:
                A2.append(c)
        
        q = [uniforme(qMin, qMax) for c in range(C)]
        q0 = sum(q[c] for c in A1)
        d = [[uniforme(dMin, dMax) for i in range(g)] for t in range(H)]
        d[0] = [d[0][0]]

        viavel = q0 + s0 >= d[0][0]

    ca = [uniforme(caMin, caMax) for c in range(C)]
    cc = [uniforme(ccMin, ccMax) for c in range(C)]
    cp = [uniforme(cpMin, cpMax) for c in range(C)]

    # Probabilidades dos cenários por estágio
    alfa = 4 / g
    beta = 4 - alfa
    p = []
    p.append([1])
    for t in range(1, H):
        p.append([0 for i in range(g)])
        falhou = True
        while falhou:
            falhou = False
            p[t][g-1] = 1
            for i in range(g - 1):
                p[t][i] = random.beta(alfa, beta)
                p[t][g-1] -= p[t][i]
                if p[t][g-1] <= 0:
                    falhou = True
                    print(f"FALHOU: {p[t]}")
                    break
    
    # Escreve o arquivo para o SDDiP
    f = open(f"sddip-{id}-{H}-{g}-{A}-{P}.dat", 'w')
    escreveSet(f, 'C', range(1, C + 1))
    f.write('\n')
    escreveParamSet(f, 'q', q)
    escreveParamSet(f, 'ca', ca)
    escreveParamSet(f, 'cc', cc)
    escreveParamSet(f, 'cp', cp)
    escreveParam(f, 'sMin', sMin)
    escreveParam(f, 'sMax', sMax)
    for t in range(H):
        f.write(f"\nnamespace t{t}\n{{\n")
        if t < 2:
            if t == 0:
                escreveSet(f, 'A', A2, 4)
                escreveSet(f, 'AAnt', A1, 4)
            else:
                escreveSet(f, 'AAnt', A2, 4)
        elif t == 2:
            escreveSet(f, 'A2Ant', A2, 4)
        if t < H - 1:
            escreveSet(f, 'P', range(A + 1, C + 1), 4)
        if t > 0:
            escreveSet(f, 'PAnt', range(A + 1, C + 1), 4)
        escreveSet(f, 'S', [1] if t == 0 else range(1, g + 1), 4)
        f.write('\n')
        escreveParamSet(f, 'p', p[t], 4)
        escreveParamSet(f, 'd', d[t], 4)
        escreveParam(f, 'h', h, 4)
        if t == 0:
            escreveParam(f, 's0', s0, 4)
        f.write('}\n')
    f.close()
    
    # Escreve o arquivo para o SDDiP


if __name__ == '__main__':
    id = int(sys.argv[1])       # identificador da instância
    H = int(sys.argv[2])        # número de estágios
    g = int(sys.argv[3])        # número de cenários por estágio (aridade da árvore)
    A = int(sys.argv[4])        # número de cargas já adquiridas
    P = int(sys.argv[5])        # número de cargas que podem ser adquiridas
    gera(id, H, g, A, P)
