"""Small CNN: 4 blocos Conv → BatchNorm → ReLU → MaxPool, depois GlobalAveragePooling (ou Flatten), Dropout e Dense.

Só a arquitetura: o treino está em src/training/treino.py, e as otimizações de cada teste ficam no notebook.
Saída em logits; a perda aplica o softmax.
"""
import keras
from keras import layers


def small_cnn(entrada, n_classes=4, dropout=0.3, filtros=(16, 32, 64, 64), achatar=False):
    # entrada = forma de um paciente, sem o lote: (133, 134, 3) na Hilbert; com 2 dimensões, (17818, 3), vira Conv1D
    conv, pool, gap = ((layers.Conv2D, layers.MaxPooling2D, layers.GlobalAveragePooling2D) if len(entrada) == 3
                       else (layers.Conv1D, layers.MaxPooling1D, layers.GlobalAveragePooling1D))
    x = entrada_ = keras.Input(entrada)
    for f in filtros:
        x = conv(f, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)  # = momentum 0,1 do torch; o 0,99 padrão do Keras demora a assentar
        x = layers.ReLU()(x)
        x = pool(2)(x)
    x = layers.Dropout(dropout)((layers.Flatten() if achatar else gap())(x))  # achatar: R9 da PARTE C (saída por posição)
    return keras.Model(entrada_, layers.Dense(n_classes)(x))  # logits; a perda aplica o softmax
