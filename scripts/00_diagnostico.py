#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_diagnostico.py -- descobre o que ESTA placa consegue fazer.

Nao tem TODO. E uma ferramenta, nao um exercicio.

Na aula (Etapa 0), cada grupo roda:
    python3 00_diagnostico.py

Quem estiver preparando as placas antes da aula roda:
    python3 00_diagnostico.py --construir
que faz a mesma coisa e AINDA constroi as engines do TensorRT, deixando-as em
cache. Isso demora -- e exatamente por demorar e que vale fazer antes.

Escreve tambem um arquivo diagnostico_<hostname>.txt com o mesmo conteudo.
"""

import os
import platform
import sys
import time

LINHA = "=" * 68

CAMINHO_FP32 = "modelo/mobilenetv2-12.onnx"
CAMINHO_FP16 = "modelo/mobilenetv2-12-fp16.onnx"
CAMINHO_LABELS = "modelo/labels.json"
PASTA_CACHE_TRT = "trt_cache"

# (apelido, arquivo do modelo, provider principal, precisao do motor)
MOTORES = [
    ("cpu",    CAMINHO_FP32, "CPUExecutionProvider",       None),
    ("cpu16",  CAMINHO_FP16, "CPUExecutionProvider",       None),
    ("cuda",   CAMINHO_FP32, "CUDAExecutionProvider",      None),
    ("cuda16", CAMINHO_FP16, "CUDAExecutionProvider",      None),
    ("trt",    CAMINHO_FP32, "TensorrtExecutionProvider",  "fp32"),
    ("trt16",  CAMINHO_FP32, "TensorrtExecutionProvider",  "fp16"),
]

saida_acumulada = []


def diz(texto=""):
    print(texto)
    saida_acumulada.append(texto)


def checa_arquivos():
    diz("[1] ARQUIVOS")
    tudo_ok = True
    for caminho in (CAMINHO_FP32, CAMINHO_FP16, CAMINHO_LABELS):
        existe = os.path.isfile(caminho)
        tamanho = os.path.getsize(caminho) / 1e6 if existe else 0
        diz("    %-36s %s  %.1f MB"
            % (caminho, "OK   " if existe else "FALTA", tamanho))
        if not existe and caminho != CAMINHO_FP16:
            tudo_ok = False
    n_imgs = len([f for f in os.listdir("imgs")]) if os.path.isdir("imgs") else 0
    diz("    %-36s %s  %d arquivos"
        % ("imgs/", "OK   " if n_imgs >= 4 else "FALTA", n_imgs))
    if n_imgs < 4:
        tudo_ok = False
    return tudo_ok


def checa_ambiente():
    diz("")
    diz("[2] AMBIENTE")
    diz("    python           : %s" % platform.python_version())
    diz("    maquina          : %s" % platform.machine())
    try:
        import numpy
        diz("    numpy            : %s" % numpy.__version__)
    except Exception as erro:
        diz("    numpy            : FALTA (%s)" % erro)
        return None
    try:
        import cv2
        diz("    opencv           : %s" % cv2.__version__)
    except Exception as erro:
        diz("    opencv           : FALTA (%s)" % erro)
        return None
    try:
        import onnxruntime as ort
    except Exception as erro:
        diz("    onnxruntime      : FALTA (%s)" % erro)
        return None
    diz("    onnxruntime      : %s" % ort.__version__)
    diz("")
    diz("    providers compilados neste wheel:")
    for p in ort.get_available_providers():
        diz("      - %s" % p)
    return ort


def checa_api_provider_options(ort):
    """Descobre QUAL forma de passar opcoes o wheel aceita.

    Isso decide como o TODO do cache de engine precisa ser escrito.
    """
    diz("")
    diz("[3] FORMA DA API DE OPCOES DO TENSORRT")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        diz("    TensorRT nao esta neste wheel -- nao se aplica.")
        return "indisponivel"
    try:
        ort.InferenceSession(
            CAMINHO_FP32,
            providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"trt_engine_cache_enable": True,
                               "trt_engine_cache_path": os.path.join(
                                   PASTA_CACHE_TRT, "_teste")}, {}])
        diz("    provider_options=[...]  ACEITO")
        diz("    -> o TODO do cache se escreve como dicionario (forma A).")
        return "dicionario"
    except TypeError:
        diz("    provider_options=[...]  RECUSADO (TypeError)")
        diz("    -> este wheel e mais antigo: as opcoes vao por variavel de")
        diz("       ambiente (forma B). Exporte antes de rodar o lab:")
        diz("         export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1")
        diz("         export ORT_TENSORRT_CACHE_PATH=trt_cache")
        diz("         export ORT_TENSORRT_FP16_ENABLE=1   # so para o trt16")
        return "ambiente"
    except Exception as erro:
        diz("    erro inesperado: %s" % erro)
        return "erro"


def receita_diagnostico(arquivo, provider, precisao):
    if provider == "CPUExecutionProvider":
        return arquivo, ["CPUExecutionProvider"], [{}]
    if provider == "CUDAExecutionProvider":
        return arquivo, ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]
    opcoes = {"trt_engine_cache_enable": True,
              "trt_engine_cache_path": os.path.join(PASTA_CACHE_TRT,
                                                    "trt16" if precisao == "fp16" else "trt")}
    if precisao == "fp16":
        opcoes["trt_fp16_enable"] = True
    caminho = opcoes["trt_engine_cache_path"]
    if not os.path.isdir(caminho):
        os.makedirs(caminho)
    return (arquivo,
            ["TensorrtExecutionProvider", "CUDAExecutionProvider",
             "CPUExecutionProvider"],
            [opcoes, {}, {}])


def testa_motores(ort, construir):
    import numpy as np

    diz("")
    diz("[4] MOTORES")
    if construir:
        diz("    modo --construir: as engines do TensorRT serao COMPILADAS.")
        diz("    Isso pode levar varios minutos por engine. Nao interrompa.")
    diz("")
    diz("    %-8s %-30s %10s %10s  %s"
        % ("motor", "provider usado", "sessao(s)", "infer(ms)", "veredito"))
    diz("    " + "-" * 76)

    compilados = ort.get_available_providers()
    disponiveis = []
    entrada_falsa = np.zeros((1, 3, 224, 224), dtype=np.float32)

    for apelido, arquivo, provider, precisao in MOTORES:
        if not os.path.isfile(arquivo):
            diz("    %-8s %-30s %10s %10s  %s"
                % (apelido, "-", "-", "-", "modelo ausente"))
            continue
        if provider not in compilados:
            diz("    %-8s %-30s %10s %10s  %s"
                % (apelido, "-", "-", "-", "provider nao compilado"))
            continue
        if provider == "TensorrtExecutionProvider" and not construir:
            # sem --construir, so tenta se ja houver cache: evita travar a aula
            pasta = os.path.join(PASTA_CACHE_TRT,
                                 "trt16" if precisao == "fp16" else "trt")
            tem_cache = os.path.isdir(pasta) and len(os.listdir(pasta)) > 0
            if not tem_cache:
                diz("    %-8s %-30s %10s %10s  %s"
                    % (apelido, "-", "-", "-",
                       "sem cache: use --construir"))
                continue

        caminho, providers, opcoes = receita_diagnostico(arquivo, provider, precisao)
        try:
            t0 = time.time()
            try:
                sessao = ort.InferenceSession(caminho, providers=providers,
                                              provider_options=opcoes)
            except TypeError:
                sessao = ort.InferenceSession(caminho, providers=providers)
            sessao_s = time.time() - t0

            nome = sessao.get_inputs()[0].name
            tipo = sessao.get_inputs()[0].type
            dado = entrada_falsa
            if "float16" in tipo:
                dado = entrada_falsa.astype(np.float16)

            sessao.run(None, {nome: dado})          # aquece
            t0 = time.time()
            sessao.run(None, {nome: dado})
            infer_ms = (time.time() - t0) * 1000.0

            usado = sessao.get_providers()[0]
            if usado != provider:
                veredito = "CAIU PARA %s" % usado
            else:
                veredito = "OK"
                disponiveis.append(apelido)
            diz("    %-8s %-30s %10.2f %10.1f  %s"
                % (apelido, usado, sessao_s, infer_ms, veredito))
        except Exception as erro:
            texto = str(erro).replace("\n", " ")[:34]
            diz("    %-8s %-30s %10s %10s  ERRO: %s"
                % (apelido, "-", "-", "-", texto))

    return disponiveis


def veredito(disponiveis):
    diz("")
    diz("[5] VEREDITO PARA ESTA PLACA")
    if not disponiveis:
        diz("    NENHUM motor funcionou. Confira os itens [1] e [2] acima.")
        return

    diz("    motores utilizaveis: %s" % ", ".join(disponiveis))
    diz("")
    tem_gpu = any(m.startswith("cuda") for m in disponiveis)
    tem_trt = any(m.startswith("trt") for m in disponiveis)

    if tem_trt:
        diz("    -> Bloco B completo: Etapas 5, 6 e 7.")
    elif tem_gpu:
        diz("    -> Bloco B sem TensorRT: faca as Etapas 5 e 7 normalmente e")
        diz("       pule a Etapa 6 (ela vira desafio extraclasse).")
    else:
        diz("    -> Sem GPU nesta placa: rode 'cpu' e 'cpu16'. O Bloco B")
        diz("       inteiro continua valido, so que comparando modelos em vez")
        diz("       de motores. As Etapas 5 e 7 funcionam sem alteracao.")


def main():
    construir = "--construir" in sys.argv

    diz(LINHA)
    diz("DIAGNOSTICO DA PLACA  |  %s" % platform.node())
    diz(LINHA)

    if not checa_arquivos():
        diz("")
        diz("Arquivos essenciais faltando. Pare aqui e resolva antes de seguir.")
        salva()
        return 1

    ort = checa_ambiente()
    if ort is None:
        diz("")
        diz("Ambiente incompleto. Pare aqui.")
        salva()
        return 1

    checa_api_provider_options(ort)
    disponiveis = testa_motores(ort, construir)
    veredito(disponiveis)

    diz("")
    diz(LINHA)
    salva()
    return 0


def salva():
    nome = "diagnostico_%s.txt" % platform.node()
    try:
        with open(nome, "w") as f:
            f.write("\n".join(saida_acumulada) + "\n")
        print("")
        print("Relatorio salvo em: %s" % nome)
    except Exception as erro:
        print("(nao consegui salvar o relatorio: %s)" % erro)


if __name__ == "__main__":
    sys.exit(main())
