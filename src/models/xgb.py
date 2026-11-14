"""XGBoost com peso por classe (nome xgb.py para não esconder o pacote xgboost)."""
from xgboost import XGBClassifier

# ponytail: importar o treino carrega o TensorFlow (~0,5 GB) só pela pesos_por_classe(); separar se o XGBoost rodar sem rede
from src.training.treino import pesos_por_classe


def treinar_xgboost(x_tr, y_tr, x_te, seed):
    # x: (n, p); devolve as probabilidades de x_te, (n_te, 4)
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                        tree_method="hist", random_state=seed, n_jobs=-1)
    return clf.fit(x_tr, y_tr, sample_weight=pesos_por_classe(y_tr)[y_tr]).predict_proba(x_te)
