"""Treino básico das redes Keras do projeto (Small CNN, MLP): a parte que não muda entre os testes.

O notebook passa a rede (construir), o otimizador e a perda; as otimizações de cada teste (AdamW, Group DRO, focal)
ficam nas células. A ordem das chamadas é a do 04_models até 2026-11-12, porque ela fixa os pesos iniciais.
"""
import gc

import keras
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

tf.config.experimental.enable_op_determinism()  # mesma semente, mesmo resultado (a 1ª célula do notebook também chama)


def pesos_por_classe(y_, n_classes=4):
    contagem = np.bincount(y_, minlength=n_classes)
    return len(y_) / (n_classes * contagem)


class F1MacroVal(keras.callbacks.Callback):
    # põe o F1 macro da validação interna nos logs; vem antes do EarlyStopping na lista, que lê o mesmo dicionário.
    # relatar(callback, época, logs, previsões da validação): o log de cada época, escrito pelo notebook
    def __init__(self, x, y, relatar=None):
        super().__init__()
        self.x, self.y, self.relatar, self.f1s = x, y, relatar, []

    def on_epoch_end(self, epoch, logs=None):
        yp = self.model.predict(self.x, verbose=0).argmax(axis=1)
        self.f1s.append(f1_score(self.y, yp, average="macro"))
        logs["val_f1_macro"] = self.f1s[-1]
        if self.relatar:
            self.relatar(self, epoch, logs, yp)


def treinar_rede(x_tr, y_tr, x_te, seed, construir, otimizador, perda, pesos=True, epocas=60, ajuste=None,
                 relatar=None, guardar=None):
    """Treina uma rede e devolve (probabilidades de x_te, F1 macro da validação interna em cada época).

    construir(forma, n_ajuste) → keras.Model com logits; otimizador() e perda() montam o otimizador e a perda;
    pesos: peso por classe no fit; ajuste(idx) → idx: reduz a parte de ajuste (R22); guardar: caminho do modelo.
    """
    keras.backend.clear_session()  # sem isso, cada modelo novo acumula memória no TensorFlow
    gc.collect()
    tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(x_tr, y_tr))
    if ajuste is not None:  # a mesma validação; só a parte de ajuste muda
        tr = ajuste(tr)
    keras.utils.set_random_seed(seed)
    modelo = construir(x_tr.shape[1:], len(tr))
    modelo.compile(otimizador(), perda())
    f1_val = F1MacroVal(x_tr[va], y_tr[va], relatar)
    parada = keras.callbacks.EarlyStopping("val_f1_macro", mode="max", patience=15, restore_best_weights=True)
    pesos_fit = dict(enumerate(pesos_por_classe(y_tr[tr]))) if pesos else None
    modelo.fit(x_tr[tr], y_tr[tr], epochs=epocas, batch_size=32, class_weight=pesos_fit, callbacks=[f1_val, parada],
               verbose=0)
    if guardar:
        modelo.save(guardar)  # bloco C do 05_eval: o Grad-CAM precisa do modelo treinado
    return softmax(modelo.predict(x_te, verbose=0), axis=1), f1_val.f1s
