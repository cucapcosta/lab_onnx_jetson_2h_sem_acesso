#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_tensorrt.py
Etapa 6: o motor que COMPILA. O custo de construir, e o cache que o evita.

Uso:
    python3 05_tensorrt.py trt        # FP32
    python3 05_tensorrt.py trt16      # FP16
    python3 05_tensorrt.py trt trt16
    python3 05_tensorrt.py trt16 --quente   # usa o cache pronto, nao reconstroi
"""

import shutil
import sys

import onnxruntime as ort

import motores


def roda(motor, limpar_cache):
    print("=" * 74)
    print("MOTOR: %s   (cache %s)"
          % (motor, "LIMPO -- vai construir do zero" if limpar_cache else "preservado"))
    print("=" * 74)

    if not motores.disponivel(motor):
        print("  indisponivel neste wheel. providers compilados:")
        print("   ", ort.get_available_providers())
        return None

    if limpar_cache:
        caminho = motores.receita(motor)[2][0].get("trt_engine_cache_path")
        if caminho:
            shutil.rmtree(caminho, ignore_errors=True)

    sessao, usado, construcao_s = motores.cria_sessao(motor)
    print("  provider realmente usado : %s" % usado)
    print("  tempo ate a sessao pronta: %.2f s" % construcao_s)

    if usado != "TensorrtExecutionProvider" and motor.startswith("trt"):
        print("  >>> ATENCAO: voce pediu TensorRT e nao obteve. Os numeros")
        print("      abaixo NAO sao do TensorRT. Investigue antes de anotar.")

    rotulos = motores.carrega_rotulos()
    tensores = motores.carrega_tensores()
    nome_entrada = sessao.get_inputs()[0].name

    t = motores.mede(sessao, nome_entrada, tensores[0][1])
    print("  1a inferencia: %8.1f ms" % t["primeira_ms"])
    print("  mediana      : %8.1f ms" % t["mediana_ms"])
    print("  p95          : %8.1f ms" % t["p95_ms"])
    print("")

    print("  %-16s %-24s %8s %8s" % ("imagem", "top-1", "conf%", "margem"))
    for linha in motores.avalia(sessao, nome_entrada, tensores, rotulos):
        print("  %-16s %-24s %7.2f%% %7.2f" % (linha["imagem"], linha["top1"],
                                               linha["conf1"], linha["margem"]))
    print("")
    return {"construcao_s": construcao_s, "tempo": t}


def main():
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    so_quente = "--quente" in sys.argv        # pula a construcao do zero
    pedidos = argumentos or ["trt", "trt16"]

    for motor in pedidos:
        frio = None
        if not so_quente:
            # com o cache limpo -- este e o custo de CONSTRUIR.
            frio = roda(motor, limpar_cache=True)
        # Sem limpar -- este e o custo de CARREGAR do cache.
        quente = roda(motor, limpar_cache=False)

        # ================================================================
        # TODO 14 -- O numero que justifica o cache.
        #
        # Com "frio" e "quente" em maos, imprima:
        #   - segundos para CONSTRUIR a engine do zero
        #   - segundos para CARREGAR a engine do cache
        #   - a razao entre os dois
        #   - as duas medianas de inferencia, lado a lado
        #
        # Pronto quando: as duas primeiras linhas diferem por ordens de
        #                grandeza e as duas ultimas sao praticamente iguais.
        #                Voce precisa saber explicar por que.
        # ================================================================
        if frio and quente:
            pass  # seu codigo aqui


if __name__ == "__main__":
    main()
