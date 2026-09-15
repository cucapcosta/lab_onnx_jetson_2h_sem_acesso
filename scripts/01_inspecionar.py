#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_inspecionar.py
Etapa 1 -- Anatomia do modelo.

Voce nao vai adivinhar como alimentar este modelo. Voce vai PERGUNTAR a ele.

Rode com:  python3 01_inspecionar.py
"""

import onnxruntime as ort

CAMINHO_MODELO = "modelo/mobilenetv2-12.onnx"


def main():
    print("=" * 62)
    print("onnxruntime versao:", ort.__version__)
    print("providers compilados neste wheel:")
    for p in ort.get_available_providers():
        print("   -", p)
    print("=" * 62)

    # ========================================================================
    # TODO 1 -- Abra o modelo criando uma InferenceSession usando
    #           explicitamente o CPUExecutionProvider.
    #
    # Pronto quando: o script roda ate o fim sem excecao e a linha
    #                "provider realmente em uso" imprime ['CPUExecutionProvider'].
    # ========================================================================
    sessao = None  # <-- substitua

    print("provider realmente em uso:", sessao.get_providers())
    print("")

    # ========================================================================
    # TODO 2 -- Imprima, para CADA entrada e CADA saida do modelo:
    #           o nome, o shape e o tipo.
    #
    #           Dica: sessao.get_inputs() e sessao.get_outputs() devolvem
    #           listas de objetos com os atributos .name, .shape e .type
    #
    # Pronto quando: voce consegue preencher a tabela do enunciado sem
    #                consultar nenhum site.
    # ========================================================================

    print("ENTRADAS")
    # seu codigo aqui

    print("SAIDAS")
    # seu codigo aqui

    # ========================================================================
    # PARE E RESPONDA (anote, vai valer na Etapa 2):
    #
    #   a) Qual das quatro dimensoes da entrada NAO e um numero? O que isso
    #      significa na pratica?
    #   b) O shape da entrada e [?, 3, 224, 224]. O OpenCV, quando le uma
    #      imagem, devolve um array com shape (altura, largura, 3).
    #      As duas coisas batem? Se nao, o que precisa acontecer?
    #   c) A saida tem 1000 posicoes. Elas ja sao probabilidades?
    #      O arquivo .onnx te diz isso em algum lugar?
    # ========================================================================


if __name__ == "__main__":
    main()
