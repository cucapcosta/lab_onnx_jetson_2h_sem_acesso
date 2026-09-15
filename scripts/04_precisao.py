#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_precisao.py
Etapa 5: metade dos bits. O que se ganha, o que se paga.

Aqui nada e compilado e nada depende do TensorRT: a unica coisa que muda e
QUAL ARQUIVO .onnx voce carrega. Por isso esta etapa roda em qualquer placa.

Uso:
    python3 04_precisao.py                 # escolhe sozinho o melhor par
    python3 04_precisao.py cpu cpu16
    python3 04_precisao.py cuda cuda16
"""

import sys

import motores


def escolhe_par():
    """Prefere comparar na GPU; cai para a CPU se nao houver."""
    if motores.disponivel("cuda") and motores.disponivel("cuda16"):
        return ["cuda", "cuda16"]
    return ["cpu", "cpu16"]


def main():
    pedidos = sys.argv[1:] or escolhe_par()

    rotulos = motores.carrega_rotulos()
    tensores = motores.carrega_tensores()

    print("=" * 78)
    print("ETAPA 5 -- o mesmo modelo, dois arquivos, duas precisoes")
    print("=" * 78)
    print("")
    print("%-8s %-28s %8s %10s %10s %10s"
          % ("motor", "provider usado", "MB", "sessao(s)", "1a (ms)", "mediana"))
    print("-" * 78)

    coletado = {}
    for motor in pedidos:
        if not motores.disponivel(motor):
            print("%-8s %-28s   (indisponivel nesta placa)" % (motor, "-"))
            continue
        sessao, usado, construcao_s = motores.cria_sessao(motor)
        nome_entrada = sessao.get_inputs()[0].name
        tempo = motores.mede(sessao, nome_entrada, tensores[0][1])
        coletado[motor] = {
            "tempo": tempo,
            "avaliacao": motores.avalia(sessao, nome_entrada, tensores, rotulos),
        }
        print("%-8s %-28s %8.2f %10.2f %10.1f %10.1f"
              % (motor, usado, motores.tamanho_mb(motor), construcao_s,
                 tempo["primeira_ms"], tempo["mediana_ms"]))

    if len(coletado) < 2:
        print("")
        print("Preciso de dois motores para comparar. Rode o 00_diagnostico.py.")
        return

    a, b = [m for m in pedidos if m in coletado][:2]

    # ---------------------------------------------------------------- tamanho
    print("")
    print("TAMANHO  : %.2f MB -> %.2f MB   (%.2fx menor)"
          % (motores.tamanho_mb(a), motores.tamanho_mb(b),
             motores.tamanho_mb(a) / motores.tamanho_mb(b)))

    # --------------------------------------------------------------- latencia
    razao = coletado[a]["tempo"]["mediana_ms"] / coletado[b]["tempo"]["mediana_ms"]
    print("LATENCIA : %.1f ms -> %.1f ms   (%.2fx)"
          % (coletado[a]["tempo"]["mediana_ms"],
             coletado[b]["tempo"]["mediana_ms"], razao))
    if razao < 1.0:
        print("")
        print("           >>> O modelo menor ficou MAIS LENTO.")
        print("           >>> Nao e engano de medicao. Descubra por que.")

    # --------------------------------------------------------------- resposta
    print("")
    print("RESPOSTA (e aqui que o Bloco A volta a ser util)")
    print("%-16s %-22s %8s %8s | %-22s %8s %8s | %s"
          % ("imagem", a, "conf%", "margem", b, "conf%", "margem", "trocou"))
    print("-" * 118)
    ref = {l["imagem"]: l for l in coletado[a]["avaliacao"]}
    trocas = 0
    for linha in coletado[b]["avaliacao"]:
        r = ref[linha["imagem"]]
        trocou = linha["indice1"] != r["indice1"]
        if trocou:
            trocas += 1
        print("%-16s %-22s %7.2f%% %8.2f | %-22s %7.2f%% %8.2f | %s"
              % (linha["imagem"], r["top1"], r["conf1"], r["margem"],
                 linha["top1"], linha["conf1"], linha["margem"],
                 "SIM" if trocou else "nao"))

    print("-" * 118)
    print("concordancia de top-1: %d/%d"
          % (len(coletado[b]["avaliacao"]) - trocas, len(coletado[b]["avaliacao"])))
    menor_margem = min(l["margem"] for l in coletado[a]["avaliacao"])
    print("menor margem do conjunto (em %s): %.2f pontos percentuais" % (a, menor_margem))
    print("")
    print("PERGUNTA: concordancia total prova que o modelo menor e seguro?")
    print("          Olhe a menor margem antes de responder.")


if __name__ == "__main__":
    main()
