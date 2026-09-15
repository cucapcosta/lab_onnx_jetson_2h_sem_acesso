#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_material.py -- RODA NO COMPUTADOR.
NAO roda na Jetson.

Monta a pasta material_lab_onnx/ com TUDO o que precisa estar em ~/lab_onnx
em cada uma das 6 placas:
  modelo/mobilenetv2-12.onnx        FP32, ImageNet, 1000 classes
  modelo/mobilenetv2-12-fp16.onnx   o mesmo modelo em FP16 (gerado aqui)
  modelo/labels.json                os 1000 rotulos, em ordem de indice
  imgs/*.jpg                        5 imagens de teste
  *.py                              os scripts do lab

A placa não precisa baixar, instalar nem converter nada. Tudo o que pode
ser pre-computado foi feito aqui.

Uso:
    pip install onnx onnxconverter-common     # so para gerar o FP16
    python3 preparar_material.py
    # depois, para cada placa: pendrive, ou
    #   rsync -av material_lab_onnx/ jetson@<ip>:~/lab_onnx/

Requer: internet e Python 3. Nao requer PyTorch (ver OPCAO B no final).
"""

import json
import shutil
import os
import sys

try:
    from urllib.request import urlopen
except ImportError:  # Python 2
    sys.exit("Rode com python3.")

DESTINO = "material_lab_onnx"

# OPCAO A (padrao): baixa o MobileNetV2 ja exportado do ONNX Model Zoo (opset 12).
# E o mesmo modelo do torchvision convertido, entao a normalizacao ImageNet vale.
URL_MODELO = ("https://media.githubusercontent.com/media/onnx/models/main/"
              "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx")

URL_LABELS = ("https://raw.githubusercontent.com/anishathalye/"
              "imagenet-simple-labels/master/imagenet-simple-labels.json")

# Imagens de teste. Foram escolhidas porque DISCRIMINAM erros de pre-processamento:
# cada uma quebra de um jeito diferente quando o pre-processamento esta errado.
# (As respostas esperadas estao no gabarito.)
IMAGENS = {
    "babuino.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/baboon.jpg",
    "futebol.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg",
    "frutas.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/fruits.jpg",
    "cachorro.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    # A "imagem fragil". So e usada no Bloco B. Margem entre top-1 e top-2 de
    # apenas ~8 pontos percentuais: e a candidata a trocar de resposta no FP16.
    "onibus.jpg": ("https://raw.githubusercontent.com/ultralytics/ultralytics/"
                   "main/ultralytics/assets/bus.jpg"),
}


def baixar(url, caminho):
    if os.path.exists(caminho) and os.path.getsize(caminho) > 1024:
        print("  ja existe: %s" % caminho)
        return
    print("  baixando %s ..." % os.path.basename(caminho))
    dados = urlopen(url, timeout=180).read()
    with open(caminho, "wb") as f:
        f.write(dados)
    print("    %.1f MB" % (len(dados) / 1e6))


def gerar_fp16(caminho_fp32, caminho_fp16):
    """Gera a versao FP16 do modelo.

    keep_io_types=True mantem entrada e saida em float32: a interface do
    modelo nao muda, so o interior. E por isso que o mesmo pre-processamento
    do Bloco A serve para os dois arquivos.
    """
    if os.path.exists(caminho_fp16) and os.path.getsize(caminho_fp16) > 1024:
        print("  ja existe: %s" % caminho_fp16)
        return True
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("  !! falta pacote. Rode:")
        print("     pip install onnx onnxconverter-common")
        print("  O lab funciona sem o FP16, mas a Etapa 5 perde metade da graca.")
        return False
    print("  convertendo para FP16 ...")
    modelo = onnx.load(caminho_fp32)
    modelo16 = float16.convert_float_to_float16(modelo, keep_io_types=True)
    onnx.save(modelo16, caminho_fp16)
    print("    %.2f MB -> %.2f MB"
          % (os.path.getsize(caminho_fp32) / 1e6,
             os.path.getsize(caminho_fp16) / 1e6))
    return True


def main():
    os.makedirs(os.path.join(DESTINO, "modelo"), exist_ok=True)
    os.makedirs(os.path.join(DESTINO, "imgs"), exist_ok=True)

    print("1) modelo FP32")
    caminho_fp32 = os.path.join(DESTINO, "modelo", "mobilenetv2-12.onnx")
    baixar(URL_MODELO, caminho_fp32)

    print("1b) modelo FP16")
    gerar_fp16(caminho_fp32,
               os.path.join(DESTINO, "modelo", "mobilenetv2-12-fp16.onnx"))

    print("2) rotulos")
    baixar(URL_LABELS, os.path.join(DESTINO, "modelo", "labels.json"))
    with open(os.path.join(DESTINO, "modelo", "labels.json")) as f:
        rotulos = json.load(f)
    assert len(rotulos) == 1000, "labels.json deveria ter 1000 entradas"
    print("  ok: 1000 rotulos, indice 0 = %s" % rotulos[0])

    print("3) imagens de teste")
    for nome, url in sorted(IMAGENS.items()):
        baixar(url, os.path.join(DESTINO, "imgs", nome))

    print("")
    print("4) scripts do lab")
    origem_scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "scripts")
    if os.path.isdir(origem_scripts):
        for nome in sorted(os.listdir(origem_scripts)):
            if nome.endswith(".py"):
                shutil.copy(os.path.join(origem_scripts, nome),
                            os.path.join(DESTINO, nome))
                print("  copiado: %s" % nome)
    else:
        print("  !! pasta scripts/ nao encontrada ao lado deste arquivo;")
        print("     copie os .py para '%s/' na mao." % DESTINO)

    tamanho = sum(os.path.getsize(os.path.join(raiz, f))
                  for raiz, _, arqs in os.walk(DESTINO) for f in arqs)
    print("")
    print("Pronto. '%s/' tem %.1f MB -- cabe em qualquer pendrive." % (DESTINO, tamanho / 1e6))
    print("")
    print("Estrutura final esperada em CADA placa, em ~/lab_onnx:")
    print("  modelo/mobilenetv2-12.onnx        modelo/mobilenetv2-12-fp16.onnx")
    print("  modelo/labels.json")
    print("  imgs/{babuino,futebol,frutas,cachorro,onibus}.jpg")
    print("  00_diagnostico.py  01_inspecionar.py  02_classificar.py")
    print("  03_benchmark.py    motores.py         04_precisao.py")
    print("  05_tensorrt.py     06_matriz.py")
    print("")
    print("Se alguem tiver acesso as placas antes da aula, mande junto o")
    print("bring_up.sh -- ele copia tudo isso e ja constroi as engines.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# OPCAO B -- exportar o modelo voce mesma, se preferir (precisa de PyTorch no PC):
#
#   import torch, torchvision
#   m = torchvision.models.mobilenet_v2(pretrained=True).eval()
#   dummy = torch.randn(1, 3, 224, 224)
#   torch.onnx.export(m, dummy, "mobilenetv2-12.onnx",
#                     input_names=["input"], output_names=["output"],
#                     dynamic_axes={"input": {0: "batch_size"},
#                                   "output": {0: "batch_size"}},
#                     opset_version=12)
#
# O contrato resultante e o mesmo usado no lab:
#   entrada "input"  float32 [batch_size, 3, 224, 224]
#   saida   "output" float32 [batch_size, 1000]   <- logits crus, SEM softmax
#
# Mantenha opset_version=12: o onnxruntime que acompanha o JetPack 4.6.x e
# relativamente antigo e opsets muito novos podem nao carregar.
# ----------------------------------------------------------------------------
