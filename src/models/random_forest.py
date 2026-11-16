"""Random Forest com parâmetros fixos da literatura (M1; WPFS, DÍAZ-URIARTE 2006, STATNIKOV 2008).

500 árvores, √p atributos por divisão, folha mínima 1, peso balanceado por classe. Só o modelo: o laço dos folds fica no notebook.
"""
from sklearn.ensemble import RandomForestClassifier


def treinar_random_forest(x_tr, y_tr, x_te, seed):
    # x: (n, p); devolve as probabilidades de x_te, (n_te, 4)
    clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced",
                                 n_jobs=-1, random_state=seed)
    return clf.fit(x_tr, y_tr).predict_proba(x_te)
