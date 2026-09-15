#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_matriz.py
Etapa 7: a matriz que fecha a aula.

Roda TODOS os motores disponiveis sobre as MESMAS 4 imagens e cruza, numa unica tabela,
o que cada um custa e o que cada um responde.

O motor de referencia e a CPU em FP32: e contra ele que a concordancia de
top-1 dos outros e medida.

Uso:
    python3 06_matriz.py
    python3 06_matriz.py cpu cuda      # subconjunto
"""

import sys

import motores


def main():
    pedidos = sys.argv[1:] or list(motores.MOTORES)

    rotulos = motores.carrega_rotulos()
    tensores = motores.carrega_tensores()

    coletado = {}
    for motor in pedidos:
        if not motores.disponivel(motor):
            print("[pulando %s: nao compilado neste wheel]" % motor)
            continue
        sessao, usado, construcao_s = motores.cria_sessao(motor)
        nome_entrada = sessao.get_inputs()[0].name
        coletado[motor] = {
            "usado": usado,
            "construcao_s": construcao_s,
            "tempo": motores.mede(sessao, nome_entrada, tensores[0][1]),
            "avaliacao": motores.avalia(sessao, nome_entrada, tensores, rotulos),
        }

    if not coletado:
        sys.exit("nenhum motor disponivel")

    # ---------------------------------------------------------------- custo
    print("")
    print("CUSTO")
    print("%-8s %-28s %7s %10s %10s %10s %10s"
          % ("motor", "provider usado", "MB", "sessao(s)", "1a (ms)", "mediana", "p95"))
    print("-" * 90)
    for motor in pedidos:
        if motor not in coletado:
            continue
        d = coletado[motor]
        print("%-8s %-28s %7.2f %10.2f %10.1f %10.1f %10.1f"
              % (motor, d["usado"], motores.tamanho_mb(motor), d["construcao_s"],
                 d["tempo"]["primeira_ms"], d["tempo"]["mediana_ms"],
                 d["tempo"]["p95_ms"]))

    # ------------------------------------------------------------- resposta
    referencia = "cpu" if "cpu" in coletado else pedidos[0]
    ref = {l["imagem"]: l for l in coletado[referencia]["avaliacao"]}

    print("")
    print("RESPOSTA  (referencia = %s)" % referencia)
    print("%-8s %-16s %-22s %8s %8s %6s"
          % ("motor", "imagem", "top-1", "conf%", "margem", "==ref?"))
    print("-" * 74)
    for motor in pedidos:
        if motor not in coletado:
            continue
        for linha in coletado[motor]["avaliacao"]:
            igual = "sim" if linha["indice1"] == ref[linha["imagem"]]["indice1"] else "NAO"
            print("%-8s %-16s %-22s %7.2f%% %7.2f %6s"
                  % (motor, linha["imagem"], linha["top1"],
                     linha["conf1"], linha["margem"], igual))

    # -------------------------------------------------------------- resumo
    print("")
    print("RESUMO")
    print("%-8s %10s %14s %16s %16s"
          % ("motor", "mediana", "aceleracao", "concordancia", "maior desvio conf"))
    print("-" * 70)
    base = coletado[referencia]["tempo"]["mediana_ms"]
    # ========================================================================
    # TODO 15 -- Feche a matriz. Para cada motor, numa unica linha:
    #
    #   mediana | aceleracao sobre a referencia | concordancia de top-1 com a
    #   referencia (quantas de quantas imagens) | maior desvio absoluto de
    #   confianca em relacao a referencia, em pontos percentuais
    #
    # As duas ultimas colunas sao o ponto da aula inteira: elas so existem
    # porque o Bloco A construiu um pipeline que sabe a resposta certa. Sem
    # ele, "2x mais rapido" seria um numero sem contrapeso.
    #
    # Compare INDICE, nao o texto do rotulo.
    #
    # Pronto quando: a tabela permite responder a pergunta impressa no fim do
    #                script SEM rodar mais nada.
    # ========================================================================
    # seu codigo aqui

    # ==================================================================
    # TODO 15 -- a pergunta que fecha a aula
    # ==================================================================
    print("")
    print("=" * 74)
    print("Escreva, em 3 linhas, qual motor voce colocaria em producao para:")
    print("  (a) um robo que classifica 30 quadros por segundo, ligado o dia todo")
    print("  (b) um script disparado por sensor, que roda 1 vez e encerra")
    print("  (c) um sistema de triagem medica")
    print("Justifique com NUMEROS desta tabela -- os tres casos nao tem a")
    print("mesma resposta, e um deles nao se decide por velocidade nenhuma.")
    print("=" * 74)


if __name__ == "__main__":
    main()
