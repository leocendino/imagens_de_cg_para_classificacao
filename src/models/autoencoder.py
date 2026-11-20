"""Autoencoder por ômica e a rede 256-128 do CustOmics (BENKIRANE 2023), usados no M3 (2026-11-17).

Pré-registro no diário, 2026-11-17, "[M3] Pré-registro". O autoencoder é ajustado sem o rótulo, nos 85% do treino do fold
(a mesma divisão do treino.treinar_rede), com parada pelo erro de reconstrução nos 15% restantes; depois, o codificador dá
os k números de cada paciente, que o notebook passa à logística (M3a) ou à rede_customics() (M3b).
"""
import gc

import keras
import numpy as np
from keras import layers
from sklearn.model_selection import StratifiedShuffleSplit


def divisao(y_tr, seed):
    # (ajuste, validação) do treino do fold: a mesma do treino.treinar_rede (só depende do rótulo, do n e da semente)
    return next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(np.zeros(len(y_tr)), y_tr))


def _bloco(x, unidades):
    # o bloco da mlp_c1() (R21): Dense → BatchNorm → LeakyReLU 0,01 → Dropout 0,2
    x = layers.Dense(unidades)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    return layers.Dropout(0.2)(x)


def _juntar(partes):
    return partes[0] if len(partes) == 1 else layers.Concatenate()(partes)


def autoencoder(blocos, k, largura=256):
    """(autoencoder, codificador). blocos = colunas de cada ômica, na ordem da entrada (DNA: [17818] * 3; RNA: [17734])."""
    entrada = keras.Input((sum(blocos),))
    inicio = np.cumsum([0, *blocos])
    ocultas = [_bloco(entrada[:, a:b], largura) for a, b in zip(inicio[:-1], inicio[1:])]  # um codificador por ômica
    gargalo = layers.Dense(k, name="gargalo")(_juntar(ocultas))                          # linear, sem BatchNorm nem dropout
    saida = _juntar([layers.Dense(b)(_bloco(gargalo, largura)) for b in blocos])          # decodificador simétrico, saída linear
    return keras.Model(entrada, saida), keras.Model(entrada, gargalo)


def treinar_autoencoder(x_tr, y_tr, x_te, seed, blocos, k, epocas=100, relatar=None):
    """Ajusta o autoencoder e devolve (k números do treino, k números do teste, informações do ajuste).

    y_tr só define a divisão de 15% (a mesma da rede base); o autoencoder não usa o rótulo.
    relatar(época, logs): o log de cada época, escrito pelo notebook (logs tem loss e val_loss).
    """
    keras.backend.clear_session()  # sem isso, cada modelo novo acumula memória no TensorFlow
    gc.collect()
    aj, va = divisao(y_tr, seed)
    keras.utils.set_random_seed(seed)
    modelo, codificador = autoencoder(blocos, k)
    modelo.compile(keras.optimizers.Adam(1e-3), "mse")
    parada = keras.callbacks.EarlyStopping("val_loss", patience=15, restore_best_weights=True)
    extra = [keras.callbacks.LambdaCallback(on_epoch_end=relatar)] if relatar else []
    h = modelo.fit(x_tr[aj], x_tr[aj], validation_data=(x_tr[va], x_tr[va]), epochs=epocas, batch_size=32,
                   callbacks=[parada, *extra], verbose=0).history
    info = {"épocas": len(h["loss"]), "melhor época": int(np.argmin(h["val_loss"])) + 1,
            "MSE teste": float(modelo.evaluate(x_te, x_te, batch_size=32, verbose=0))}
    return codificador.predict(x_tr, verbose=0), codificador.predict(x_te, verbose=0), info


def rede_customics(entrada, n_classes=4):
    # o classificador depois do gargalo no CustOmics: 256 → 128 (ReLU), com BatchNorm e dropout em cada camada; saída em logits
    x = entrada_ = keras.Input(entrada)
    for unidades in (256, 128):
        x = layers.Dense(unidades)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
    return keras.Model(entrada_, layers.Dense(n_classes)(x))
