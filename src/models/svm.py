"""SVM polinomial, um contra o resto, com o grau e o C escolhidos dentro do treino (M2; grade do STATNIKOV 2008).

Grade: grau {1, 2, 3} × C {0,01; 1; 100}, coef0 = 1, peso balanceado por classe. gamma="scale" (1 / (p × variância)) em vez de
gamma = 1: com dezenas de milhares de colunas em z, o produto interno é grande e, elevado ao grau, deixaria o C sem efeito.
A escolha é por CV interna estratificada no treino, pelo F1 macro médio; empate → o grau menor e depois o C menor (ordem da
GRADE). O escolhido é refeito no treino inteiro com probability=True (calibração de Platt interna) para as 4 probabilidades.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

GRADE = [(grau, c) for grau in (1, 2, 3) for c in (0.01, 1.0, 100.0)]


def _svm(grau, c, calibrar):
    svc = SVC(kernel="poly", degree=grau, coef0=1, gamma="scale", C=c, class_weight="balanced")
    return OneVsRestClassifier(CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False) if calibrar else svc)


def treinar_svm(x_tr, y_tr, x_te, seed, folds_internos=3):
    # seed: só a CV interna de escolha (a calibração usa folds estratificados sem sorteio)
    # devolve (probabilidades de x_te (n_te, 4), (grau, C) escolhidos, F1 macro médio da CV interna de cada (grau, C) da GRADE)
    cv = list(StratifiedKFold(folds_internos, shuffle=True, random_state=seed).split(x_tr, y_tr))
    f1s = [np.mean([f1_score(y_tr[va], _svm(grau, c, False).fit(x_tr[tr], y_tr[tr]).predict(x_tr[va]), average="macro")
                    for tr, va in cv]) for grau, c in GRADE]
    escolha = GRADE[int(np.argmax(f1s))]  # o argmax fica com o primeiro dos empatados
    return _svm(*escolha, True).fit(x_tr, y_tr).predict_proba(x_te), escolha, f1s
