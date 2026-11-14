"""MLP para dados tabulares: a rede de referência do WPFS (MARGELOIU 2023), 100-100-10, usada no R21.

Só a arquitetura: o treino está em src/training/treino.py. Saída em logits, como a small_cnn().
"""
import keras
from keras import layers


def mlp_c1(entrada, n_classes=4):
    # entrada = (p,): um paciente como vetor; no R21, (53454,) = 3 canais × 17.818 genes
    x = entrada_ = keras.Input(entrada)
    for unidades in (100, 100, 10):
        x = layers.Dense(unidades)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # o mesmo momento da small_cnn()
        x = layers.LeakyReLU(negative_slope=0.01)(x)    # 0,01 = padrão do PyTorch, em que o WPFS foi escrito (o do Keras é 0,3)
        x = layers.Dropout(0.2)(x)
    return keras.Model(entrada_, layers.Dense(n_classes)(x))
