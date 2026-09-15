#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motores.py -- modulo compartilhado pelo Bloco B (Etapas 5, 6 e 7).

Um motor e a combinacao de DUAS escolhas independentes:

    qual ARQUIVO de modelo          FP32 (14 MB)  ou  FP16 (7 MB)
    qual EXECUTION PROVIDER         CPU  |  CUDA  |  TensorRT

    apelido   arquivo   provider    o que muda
    -------   -------   ---------   -----------------------------------------
    cpu       FP32      CPU         a referencia de tudo
    cpu16     FP16      CPU         so o arquivo mudou
    cuda      FP32      CUDA        so o provider mudou
    cuda16    FP16      CUDA        os dois mudaram
    trt       FP32      TensorRT    provider que COMPILA uma engine
    trt16     FP32      TensorRT    a mesma engine, compilada em FP16

Repare que trt e trt16 usam o MESMO arquivo .onnx: quem decide a precisao ali
e a flag do motor, nao o modelo. Ja cpu16/cuda16 usam outro arquivo, onde a
precisao esta gravada nos pesos. Sao dois caminhos diferentes para a mesma
ideia, e a Etapa 7 existe para compara-los.

Os dois modelos tem a MESMA interface (entrada e saida em float32), entao o
seu pre-processamento do Bloco A serve para todos os seis motores sem
nenhuma alteracao.

Voce nao roda este arquivo diretamente -- ele e importado pelos scripts 04,
05 e 06. Os TODOs 11, 12 e 13 estao aqui dentro.
"""

import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

CAMINHO_FP32 = "modelo/mobilenetv2-12.onnx"
CAMINHO_FP16 = "modelo/mobilenetv2-12-fp16.onnx"
CAMINHO_LABELS = "modelo/labels.json"
PASTA_CACHE_TRT = "trt_cache"

# As quatro do Bloco A, mais a "imagem fragil" que so entra no Bloco B.
IMAGENS = ["imgs/futebol.jpg", "imgs/babuino.jpg",
           "imgs/frutas.jpg", "imgs/cachorro.jpg",
           "imgs/onibus.jpg"]

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# (arquivo, provider principal, precisao pedida ao TensorRT)
DEFINICAO = {
    "cpu":    (CAMINHO_FP32, "CPUExecutionProvider", None),
    "cpu16":  (CAMINHO_FP16, "CPUExecutionProvider", None),
    "cuda":   (CAMINHO_FP32, "CUDAExecutionProvider", None),
    "cuda16": (CAMINHO_FP16, "CUDAExecutionProvider", None),
    "trt":    (CAMINHO_FP32, "TensorrtExecutionProvider", "fp32"),
    "trt16":  (CAMINHO_FP32, "TensorrtExecutionProvider", "fp16"),
}

MOTORES = ("cpu", "cpu16", "cuda", "cuda16", "trt", "trt16")


def preprocessa_correto(imagem_bgr):
    # COPIE aqui a sua funcao da Etapa 3, sem mudar uma virgula.
    # Ela nao muda com o motor -- e por isso que ela pode servir de
    # instrumento de medida.
    raise NotImplementedError("copie o seu preprocessa_correto da Etapa 3")


def softmax(vetor):
    # COPIE aqui a sua funcao da Etapa 3.
    raise NotImplementedError("copie o seu softmax da Etapa 3")


def carrega_rotulos():
    with open(CAMINHO_LABELS) as f:
        return json.load(f)


def carrega_tensores():
    tensores = []
    for caminho in IMAGENS:
        imagem = cv2.imread(caminho)
        if imagem is None:
            raise IOError("nao consegui abrir %s" % caminho)
        tensores.append((os.path.basename(caminho), preprocessa_correto(imagem)))
    return tensores


def tamanho_mb(motor):
    arquivo = DEFINICAO[motor][0]
    return os.path.getsize(arquivo) / 1e6 if os.path.isfile(arquivo) else 0.0


# ============================================================================
# TODO 13 -- Complete a receita do TensorRT (Etapa 6).
#
# Duas chaves de CACHE, sem as quais a engine e recompilada a cada execucao
# do script -- e na Nano isso custa minutos, toda vez:
#
#     "trt_engine_cache_enable": True
#     "trt_engine_cache_path":   <uma pasta POR MOTOR>
#
# Use PASTA_CACHE_TRT + o nome do motor. Pasta separada por motor nao e
# estilo, e requisito: FP32 e FP16 geram engines diferentes e uma
# sobrescreveria a outra.
#
# E uma chave de PRECISAO, so quando precisao == "fp16":
#
#     "trt_fp16_enable": True
#
# Se o 00_diagnostico.py disse que este wheel nao aceita provider_options,
# as mesmas opcoes vao por variavel de ambiente -- veja o relatorio dele.
# ============================================================================
def receita(motor):
    """Devolve (arquivo, lista_de_providers, lista_de_opcoes)."""
    arquivo, provider, precisao = DEFINICAO[motor]

    if provider == "CPUExecutionProvider":
        return arquivo, ["CPUExecutionProvider"], [{}]

    if provider == "CUDAExecutionProvider":
        return arquivo, ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]

    opcoes_trt = {
        # TODO 13: suas duas chaves de cache aqui
    }
    # TODO 13: a chave de FP16, so quando precisao == "fp16"

    caminho_cache = opcoes_trt.get("trt_engine_cache_path")
    if caminho_cache and not os.path.isdir(caminho_cache):
        os.makedirs(caminho_cache)

    # Lista de PRIORIDADE: o que o TensorRT nao compila cai para o CUDA, e o
    # que o CUDA nao tem cai para a CPU. Um mesmo grafo pode acabar repartido.
    return (arquivo,
            ["TensorrtExecutionProvider", "CUDAExecutionProvider",
             "CPUExecutionProvider"],
            [opcoes_trt, {}, {}])


def disponivel(motor):
    arquivo, provider, _ = DEFINICAO[motor]
    if not os.path.isfile(arquivo):
        return False
    if provider == "CPUExecutionProvider":
        return True
    return provider in ort.get_available_providers()


# ============================================================================
# TODO 12 -- Separe o custo de MONTAR o motor do custo de USAR o motor.
#
# Cronometre a criacao da InferenceSession e devolva
#     (sessao, provider_realmente_usado, segundos_de_construcao)
#
# Mantenha o try/except: wheels mais antigos nao aceitam provider_options, e
# sem o except o script quebra em vez de degradar.
#
#   arquivo, providers, opcoes = receita(motor)
#   try:
#       sessao = ort.InferenceSession(arquivo, providers=providers,
#                                     provider_options=opcoes)
#   except TypeError:
#       sessao = ort.InferenceSession(arquivo, providers=providers)
#
# Pronto quando: o 04_precisao.py imprime a coluna "sessao(s)" com numeros
#                plausiveis para os dois motores.
# ============================================================================
def cria_sessao(motor):
    # seu codigo aqui
    raise NotImplementedError("TODO 12")


def mede(sessao, nome_entrada, tensor, n_aquecimento=10, n_medicoes=30):
    """Ja vem pronto: e o seu TODO 10 da Etapa 4, com p95 a mais."""
    entradas = {nome_entrada: tensor}

    t0 = time.time()
    sessao.run(None, entradas)
    primeira_ms = (time.time() - t0) * 1000.0

    for _ in range(n_aquecimento):
        sessao.run(None, entradas)

    amostras = []
    for _ in range(n_medicoes):
        t0 = time.time()
        sessao.run(None, entradas)
        amostras.append((time.time() - t0) * 1000.0)

    amostras = np.array(amostras)
    return {
        "primeira_ms": primeira_ms,
        "mediana_ms": float(np.median(amostras)),
        "min_ms": float(amostras.min()),
        "max_ms": float(amostras.max()),
        "p95_ms": float(np.percentile(amostras, 95)),
    }


# ============================================================================
# TODO 11 -- A verificacao que o Bloco A tornou possivel (Etapa 5).
#
# Para cada imagem, devolva um dicionario com:
#     "imagem", "top1", "indice1", "conf1", "top2", "conf2", "margem"
#
# onde margem = (probabilidade do 1o) - (probabilidade do 2o), em pontos
# percentuais.
#
# Por que a margem, e nao so a confianca? Porque a margem diz o quao FRAGIL
# aquela decisao e. Uma classificacao com 99% de confianca e margem de 0,5
# ponto esta a um arredondamento de trocar de resposta.
#
# Pronto quando: a onibus.jpg aparece com margem de um digito e as outras
#                quatro com margem de dois digitos.
# ============================================================================
def avalia(sessao, nome_entrada, tensores, rotulos):
    # seu codigo aqui
    raise NotImplementedError("TODO 11")
