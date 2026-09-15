#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_classificar.py
Etapas 2 e 3 -- Pre-processar, inferir, ler a saida.

Rode com:  python3 02_classificar.py imgs/futebol.jpg
"""

import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

CAMINHO_MODELO = "modelo/mobilenetv2-12.onnx"
CAMINHO_LABELS = "modelo/labels.json"

LADO = 224
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# ESTE JA VEM PRONTO. Nao altere.
#
# Ele monta um tensor com o shape e o dtype exatos que voce descobriu na
# Etapa 1. Guarde essa frase: o shape esta certo.
# ============================================================================
def preprocessa_ingenuo(imagem_bgr):
    img = cv2.resize(imagem_bgr, (LADO, LADO))
    img = img.astype(np.float32)
    tensor = np.transpose(img, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)


# ============================================================================
# TODO 7 -- (VOCE SO CHEGA AQUI DEPOIS DA ETAPA 3)
#
# Escreva o pre-processamento correto. Compare com o ingenuo acima e
# descubra o que falta. Sao tres coisas, e nenhuma delas muda o shape.
#
# Pronto quando: as 4 imagens de teste sao classificadas corretamente
#                (o gabarito de acertos esta no enunciado).
# ============================================================================
def preprocessa_correto(imagem_bgr):
    # seu codigo aqui
    raise NotImplementedError("TODO 7")


# ============================================================================
# TODO 5 -- Implemente o softmax.
#
#   softmax(x)_i = exp(x_i) / soma_j exp(x_j)
#
# Atencao: a implementacao literal estoura em float32. Subtraia o maximo
# do vetor antes de exponenciar -- o resultado matematico e o mesmo.
#
# Pronto quando: softmax(np.array([1.0, 2.0, 3.0])) devolve algo que soma 1.0
#                e softmax(np.array([1000.0, 1001.0])) nao devolve nan.
# ============================================================================
def softmax(vetor):
    # seu codigo aqui
    raise NotImplementedError("TODO 5")


# ============================================================================
# TODO 6 -- Devolva as k maiores probabilidades como uma lista de tuplas
#           (rotulo, porcentagem), da maior para a menor.
#
#           Dica: np.argsort ordena crescente. [::-1] inverte.
#
# Pronto quando: a saida do script mostra 5 linhas ordenadas de cima para baixo.
# ============================================================================
def top_k(probabilidades, rotulos, k=5):
    # seu codigo aqui
    raise NotImplementedError("TODO 6")


def classifica(sessao, nome_entrada, rotulos, caminho_imagem, funcao_preproc):
    imagem_bgr = cv2.imread(caminho_imagem)
    if imagem_bgr is None:
        sys.exit("Nao consegui abrir a imagem: %s" % caminho_imagem)

    # ========================================================================
    # TODO 3 -- Chame a funcao de pre-processamento recebida em funcao_preproc
    #           e confira que o tensor bate com o contrato da Etapa 1.
    #
    # Pronto quando: tensor.shape == (1, 3, 224, 224) e tensor.dtype == float32
    # ========================================================================
    tensor = None  # <-- substitua

    # ========================================================================
    # TODO 4 -- Rode a inferencia e pegue o vetor de 1000 numeros.
    #
    #   sessao.run(lista_de_saidas, dicionario_de_entradas)
    #     - lista_de_saidas = None significa "me devolva todas"
    #     - o dicionario e {nome_da_entrada: tensor}
    #     - o retorno e uma LISTA de arrays, um por saida
    #     - a saida tem shape (1, 1000): o 1 da frente e o lote
    #
    # Pronto quando: logits.shape == (1000,)
    # ========================================================================
    logits = None  # <-- substitua

    probabilidades = softmax(logits)
    return tensor, top_k(probabilidades, rotulos)


def main():
    with open(CAMINHO_LABELS) as f:
        rotulos = json.load(f)

    sessao = ort.InferenceSession(CAMINHO_MODELO,
                                  providers=["CPUExecutionProvider"])
    nome_entrada = sessao.get_inputs()[0].name

    imagens = sys.argv[1:]
    if not imagens:
        imagens = ["imgs/futebol.jpg", "imgs/babuino.jpg",
                   "imgs/frutas.jpg", "imgs/cachorro.jpg"]

    # ========================================================================
    # TODO 8 -- Rode CADA imagem com os DOIS pre-processamentos e imprima as
    #           duas listas top-5 lado a lado, junto com shape, dtype e a
    #           faixa de valores do tensor (tensor.min() e tensor.max()).
    #
    #           Ate terminar a Etapa 3, deixe so o INGENUO na lista abaixo.
    #
    # Pronto quando: voce consegue preencher a tabela comparativa do enunciado
    #                para as 4 imagens.
    # ========================================================================
    lista_de_preprocessamentos = [
        ("INGENUO", preprocessa_ingenuo),
        # ("CORRETO", preprocessa_correto),   <-- descomente na Etapa 3
    ]

    for caminho in imagens:
        print("=" * 66)
        print("IMAGEM:", caminho)
        for nome_preproc, funcao in lista_de_preprocessamentos:
            tensor, resultados = classifica(sessao, nome_entrada, rotulos,
                                            caminho, funcao)
            print("")
            print("  [%s]  shape=%s dtype=%s  faixa=[%.2f, %.2f]"
                  % (nome_preproc, tensor.shape, tensor.dtype,
                     tensor.min(), tensor.max()))
            for posicao, (rotulo, pct) in enumerate(resultados, start=1):
                marca = ">>" if posicao == 1 else "  "
                print("   %s %d. %-32s %6.2f%%" % (marca, posicao, rotulo, pct))
        print("")


if __name__ == "__main__":
    main()
