#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_benchmark.py
Etapa 4 -- O mesmo .onnx, o mesmo codigo, execution providers diferentes.

Rode com:  python3 03_benchmark.py          (todos os disponiveis)
           python3 03_benchmark.py cpu
           python3 03_benchmark.py cuda
"""

import json
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

CAMINHO_MODELO = "modelo/mobilenetv2-12.onnx"
CAMINHO_LABELS = "modelo/labels.json"
IMAGEM = "imgs/babuino.jpg"

N_AQUECIMENTO = 10
N_MEDICOES = 30

APELIDOS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


def preprocessa_correto(imagem_bgr):
    # COPIE aqui a sua funcao da Etapa 3.
    raise NotImplementedError("copie o seu preprocessa_correto do 02")


# ============================================================================
# TODO 9 -- Crie a sessao pedindo o provider recebido em nome_provider, e
#           devolva TAMBEM qual provider o onnxruntime realmente colocou em uso.
#
#           Duas coisas importam aqui:
#           - a lista de providers e uma lista de PRIORIDADE, nao uma escolha
#             unica: quem nao roda num provider cai no seguinte da lista;
#           - pedir nao e obter. sessao.get_providers() conta a verdade.
#
# Pronto quando: rodando com 'cuda' numa placa sem CUDA disponivel, o script
#                avisa que caiu para CPU em vez de mentir que usou a GPU.
# ============================================================================
def cria_sessao(nome_provider):
    # seu codigo aqui
    raise NotImplementedError("TODO 9")


# ============================================================================
# TODO 10 -- Meca a latencia de inferencia e devolva uma tupla
#            (primeira_ms, mediana_ms, minimo_ms, maximo_ms).
#
#            O procedimento:
#              1. cronometre a PRIMEIRA inferencia separadamente e guarde;
#              2. rode N_AQUECIMENTO inferencias jogando o tempo fora;
#              3. rode N_MEDICOES inferencias guardando cada tempo;
#              4. devolva a primeira, a MEDIANA das medicoes, o min e o max.
#
#            Use time.time() e multiplique por 1000 para ter milissegundos.
#
#            Por que mediana e nao media? Porque uma unica interrupcao do
#            sistema operacional no meio de uma medicao estraga a media
#            inteira e nao mexe na mediana.
#
# Pronto quando: a coluna "1a (ms)" fica visivelmente maior que a mediana.
# ============================================================================
def mede(sessao, nome_entrada, tensor):
    # seu codigo aqui
    raise NotImplementedError("TODO 10")


def main():
    with open(CAMINHO_LABELS) as f:
        rotulos = json.load(f)

    imagem = cv2.imread(IMAGEM)
    if imagem is None:
        sys.exit("Nao consegui abrir %s" % IMAGEM)
    tensor = preprocessa_correto(imagem)

    disponiveis = ort.get_available_providers()
    if len(sys.argv) > 1:
        pedidos = [APELIDOS.get(a.lower(), a) for a in sys.argv[1:]]
    else:
        pedidos = [p for p in ("CPUExecutionProvider", "CUDAExecutionProvider")
                   if p in disponiveis]

    print("providers compilados neste wheel:", disponiveis)
    print("aquecimento=%d  medicoes=%d  imagem=%s"
          % (N_AQUECIMENTO, N_MEDICOES, IMAGEM))
    print("")
    print("%-26s %-26s %10s %10s %10s %10s"
          % ("PEDIDO", "USADO DE FATO", "1a (ms)", "mediana", "min", "max"))
    print("-" * 96)

    resultados = {}
    for pedido in pedidos:
        if pedido not in disponiveis:
            print("%-26s %-26s  (nao compilado neste wheel)" % (pedido, "-"))
            continue
        sessao, usado = cria_sessao(pedido)
        nome_entrada = sessao.get_inputs()[0].name
        primeira, mediana, minimo, maximo = mede(sessao, nome_entrada, tensor)
        resultados[pedido] = mediana

        alerta = "" if usado == pedido else "   <<< CAIU PARA CPU!"
        print("%-26s %-26s %10.1f %10.1f %10.1f %10.1f%s"
              % (pedido, usado, primeira, mediana, minimo, maximo, alerta))

        logits = sessao.run(None, {nome_entrada: tensor})[0][0]
        indice = int(np.argmax(logits))
        print("%-26s top-1: %s (indice %d)" % ("", rotulos[indice], indice))

    print("-" * 96)
    if "CPUExecutionProvider" in resultados and "CUDAExecutionProvider" in resultados:
        ganho = resultados["CPUExecutionProvider"] / resultados["CUDAExecutionProvider"]
        print("aceleracao CUDA sobre CPU (mediana): %.2fx" % ganho)
        print("")
        print("Agora refaca a conta usando so a PRIMEIRA inferencia de cada um.")
        print("Que conclusao voce teria tirado?")


if __name__ == "__main__":
    main()
