"""A regra do TCGA (2014) como classificador: EBV pela metilação do CDKN2A, MSI pelo número de genes mutados e GS pela
fração do genoma alterada; quem sobra das três perguntas fica CIN. Os cortes são aprendidos no treino de cada fold.
"""
import numpy as np

CLASSES = ["CIN", "MSI", "GS", "EBV"]  # = classe_idx 0, 1, 2, 3
CIN, MSI, GS, EBV = range(len(CLASSES))

QUESTOES = [  # (classe, variável, sentido, classes de treino que chegam à pergunta), na ordem da árvore do TCGA
    (EBV, "CDKN2A (beta)", +1, [CIN, MSI, GS, EBV]),
    (MSI, "genes mutados", +1, [CIN, MSI, GS]),
    (GS, "fração alterada", -1, [CIN, GS]),
]


def limiar(v, positivo):
    # corte t que dá o maior F1 da classe positiva com a regra "v > t"; candidatos = pontos médios entre valores
    u = np.unique(v)
    t = (u[:-1] + u[1:]) / 2
    prev = v[None] > t[:, None]
    f1 = 2 * (prev & positivo).sum(1) / (prev.sum(1) + positivo.sum())
    return t[f1.argmax()]


def regra_tcga(vars_, y, tr, te, questoes=QUESTOES):
    # vars_: {variável: (n,)}; y: classes (n,); tr, te: máscaras (n,). Devolve a classe prevista em te e os cortes
    yp, livre, cortes = np.full(te.sum(), CIN), np.ones(te.sum(), bool), {}
    for classe, var, sentido, grupo in questoes:
        v, aprende = sentido * vars_[var], tr & np.isin(y, grupo)
        t = limiar(v[aprende], y[aprende] == classe)
        sim = livre & (v[te] > t)
        yp[sim], livre = classe, livre & ~sim  # quem sobra das três perguntas fica CIN
        cortes[f"{CLASSES[classe]} se {var} {'>' if sentido > 0 else '<'}"] = sentido * t
    return yp, cortes
