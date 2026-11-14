"""Regressão logística multiclasse, a linha de base tabular do projeto (C = 1, peso balanceado por classe)."""
from sklearn.linear_model import LogisticRegression


def treinar_logistica(x_tr, y_tr, x_te, seed):
    # x: (n, p); devolve as probabilidades de x_te, (n_te, 4)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    return clf.fit(x_tr, y_tr).predict_proba(x_te)
