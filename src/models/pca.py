"""PCA, o controle do autoencoder no M3 (2026-11-17; pré-registro no diário, "[M3] Pré-registro")."""
import numpy as np
from sklearn.decomposition import PCA


def codigos_pca(x_tr, x_te, ajuste, k):
    """(k números do treino, k números do teste, informações). O PCA é ajustado só nos pacientes `ajuste` do treino do fold,
    os mesmos 85% em que o autoencoder ajusta os pesos (autoencoder.divisao)."""
    p = PCA(k, svd_solver="full").fit(x_tr[ajuste])  # SVD completa: determinística, sem semente
    erro = float(np.mean((p.inverse_transform(p.transform(x_te)) - x_te) ** 2))  # = o "MSE teste" do autoencoder
    return p.transform(x_tr), p.transform(x_te), {"MSE teste": erro}
