#!/bin/bash
# =============================================================================
# bring_up.sh -- preparacao de UMA Jetson Nano para o lab de ONNX Runtime.
#
# PARA QUEM TEM ACESSO FISICO AS PLACAS (tecnico de laboratorio, monitor).
# A professora nao precisa estar presente.
#
# COMO USAR
#   1. copie a pasta material_lab_onnx/ e este arquivo para a placa
#      (pendrive ou scp), em qualquer lugar
#   2. na placa:  bash bring_up.sh
#   3. devolva para a professora o arquivo relatorio_<hostname>.txt gerado
#
# O script NAO instala nada da internet e NAO altera configuracao permanente
# do sistema, exceto o swap (item 4), que pergunta antes.
# Tempo estimado: 5 a 20 minutos por placa, quase tudo no item 7.
# =============================================================================

set -u

DESTINO="$HOME/lab_onnx"
ORIGEM="$(cd "$(dirname "$0")" && pwd)"
RELATORIO="$ORIGEM/relatorio_$(hostname).txt"

exec > >(tee "$RELATORIO") 2>&1

echo "============================================================"
echo "BRING-UP  |  placa: $(hostname)  |  $(date)"
echo "============================================================"

falhas=0
aviso() { echo ""; echo ">>> $*"; echo ""; }
erro()  { echo ""; echo "!!! FALHOU: $*"; echo ""; falhas=$((falhas+1)); }

# --- 1. identificacao da placa -----------------------------------------------
echo ""
echo "[1] IDENTIFICACAO"
echo "  modelo   : $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo '?')"
echo "  jetpack  : $(cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo '?')"
echo "  kernel   : $(uname -r)"
echo "  python3  : $(python3 --version 2>&1)"
echo "  memoria  :"
free -h | sed 's/^/    /'

# --- 2. pacotes python -------------------------------------------------------
echo ""
echo "[2] PACOTES PYTHON"
python3 - <<'PY'
mods = ["numpy", "cv2", "onnxruntime"]
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print("  OK   %-14s %s" % (m, v))
    except Exception as e:
        print("  FALTA %-13s (%s)" % (m, e))
PY
python3 -c "import numpy, cv2, onnxruntime" 2>/dev/null || erro "falta pacote python essencial (veja acima)"

# --- 3. execution providers --------------------------------------------------
echo ""
echo "[3] EXECUTION PROVIDERS COMPILADOS NO WHEEL"
python3 - <<'PY'
try:
    import onnxruntime as ort
    ps = ort.get_available_providers()
    for p in ps:
        print("  -", p)
    print("")
    print("  CPU      :", "sim" if "CPUExecutionProvider" in ps else "NAO")
    print("  CUDA     :", "sim" if "CUDAExecutionProvider" in ps else "NAO  <-- Bloco B limitado")
    print("  TensorRT :", "sim" if "TensorrtExecutionProvider" in ps else "NAO  <-- Etapa 6 nao roda")
except Exception as e:
    print("  erro ao consultar:", e)
PY

# --- 4. swap -----------------------------------------------------------------
echo ""
echo "[4] SWAP  (o TensorRT precisa disso para compilar sem estourar a RAM)"
swapon --show 2>/dev/null | sed 's/^/  /'
SWAP_KB=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
SWAP_GB=$((SWAP_KB / 1024 / 1024))
echo "  total: ${SWAP_GB} GB"
if [ "$SWAP_GB" -lt 4 ]; then
  aviso "swap abaixo de 4 GB. Criar 4 GB em /var/swapfile agora? [s/N]"
  read -r resposta || resposta="n"
  if [ "$resposta" = "s" ] || [ "$resposta" = "S" ]; then
    sudo fallocate -l 4G /var/swapfile && \
    sudo chmod 600 /var/swapfile && \
    sudo mkswap /var/swapfile && \
    sudo swapon /var/swapfile && \
    echo "/var/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab && \
    echo "  swap criado e persistido no /etc/fstab" || erro "nao consegui criar swap"
  else
    echo "  pulado -- se o item 7 falhar por falta de memoria, volte aqui"
  fi
fi

# --- 5. modo de energia ------------------------------------------------------
echo ""
echo "[5] MODO DE ENERGIA  (medicoes so sao comparaveis com o clock fixo)"
sudo nvpmodel -m 0 2>/dev/null && echo "  nvpmodel -m 0 aplicado" || echo "  nvpmodel indisponivel"
sudo jetson_clocks 2>/dev/null && echo "  jetson_clocks aplicado" || echo "  jetson_clocks indisponivel"
echo "  ATENCAO: jetson_clocks nao persiste no boot. Reaplicar no dia da aula."

# --- 6. material -------------------------------------------------------------
echo ""
echo "[6] MATERIAL DO LAB"
mkdir -p "$DESTINO"
if [ -d "$ORIGEM/material_lab_onnx" ]; then
  cp -r "$ORIGEM/material_lab_onnx/." "$DESTINO/"
  echo "  copiado de material_lab_onnx/ para $DESTINO"
else
  erro "pasta material_lab_onnx/ nao encontrada ao lado deste script"
fi
echo "  conteudo de $DESTINO:"
ls -la "$DESTINO" | sed 's/^/    /'
for obrigatorio in modelo/mobilenetv2-12.onnx modelo/labels.json imgs/babuino.jpg; do
  [ -f "$DESTINO/$obrigatorio" ] || erro "faltando $obrigatorio"
done

# --- 7. diagnostico + construcao das engines ---------------------------------
echo ""
echo "[7] DIAGNOSTICO E CONSTRUCAO DAS ENGINES"
echo "    (esta e a parte demorada -- pode levar varios minutos por engine)"
cd "$DESTINO" || exit 1
if [ -f "00_diagnostico.py" ]; then
  python3 00_diagnostico.py --construir
else
  erro "00_diagnostico.py nao encontrado em $DESTINO"
fi

# --- fim ---------------------------------------------------------------------
echo ""
echo "============================================================"
if [ "$falhas" -eq 0 ]; then
  echo "BRING-UP CONCLUIDO SEM FALHAS  |  placa: $(hostname)"
else
  echo "BRING-UP CONCLUIDO COM $falhas FALHA(S)  |  placa: $(hostname)"
  echo "Procure por '!!! FALHOU' acima."
fi
echo ""
echo "Devolva este arquivo para a professora:"
echo "  $RELATORIO"
echo "============================================================"
