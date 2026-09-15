# Lab — Do `.onnx` ao motor mais rápido da Jetson


## Blocos

**Bloco A — ONNX Runtime** Você pega um arquivo `.onnx` treinado em outro computador e faz ele
classificar corretamente aqui na placa.

**Bloco B — Quantização** Você otimiza esse mesmo modelo por até seis motores diferentes, até
o limite da placa.

E aqui está a parte que importa: **o pipeline correto que você construir no Bloco A é o
instrumento que vai medir o Bloco B.** 

No fim das duas horas você deve saber responder:

1. Como descobrir o que um `.onnx` desconhecido espera receber e o que ele devolve.
2. Por que um tensor com o *shape* certo ainda assim pode produzir uma resposta errada.


---

## Etapa 0 — O que esta placa consegue fazer? 

Quais são as configurações da placa que vamos utilizar?

**1. Preparar o material** execute o arquivo preparar_material.py, copie os arquivos na pasta:

```bash
cd ~/lab_onnx
python3 preparar_material.py
```

Rode **sempre de dentro de `~/lab_onnx`** — todos os caminhos nos scripts são relativos.

```
~/lab_onnx/
├── modelo/  mobilenetv2-12.onnx   mobilenetv2-12-fp16.onnx   labels.json
├── imgs/    babuino.jpg  futebol.jpg  frutas.jpg  cachorro.jpg  onibus.jpg
├── 00_diagnostico.py   01_inspecionar.py   02_classificar.py   03_benchmark.py
├── motores.py          04_precisao.py      05_tensorrt.py      06_matriz.py
```

**2. Descubra o que a placa tem:**

```bash
python3 00_diagnostico.py
```

Leia a seção `[5] VEREDITO` e anote:

| | |
|---|---|
| versão do onnxruntime | |
| providers compilados | |
| motores utilizáveis nesta placa | |
| caminho indicado pelo veredito | |

**3. Fixe o clock**, para as medições do Bloco B serem comparáveis:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

**Olhe as imagens antes de começar.** Você precisa saber o que tem em cada uma para julgar se o
modelo acertou. As quatro primeiras são do Bloco A; a `onibus.jpg` só entra no Bloco B, e você
vai descobrir por quê.

---

# BLOCO A — ONNX Runtime

## Etapa 1 — Modelo 

Um `.onnx` não é uma caixa-preta: ele carrega a descrição das próprias entradas e saídas. A
pergunta desta etapa é **como alimentar um modelo que você nunca viu**?

Arquivo: `01_inspecionar.py`

**TODO 1** — Abra o modelo criando uma `InferenceSession` com o `CPUExecutionProvider`
explícito.
*Pronto quando:* roda sem exceção e imprime `['CPUExecutionProvider']`.

**TODO 2** — Imprima, para cada entrada e cada saída: **nome**, **shape** e **tipo**.
*Pronto quando:* você preenche a tabela abaixo sem consultar nada.

| | nome | shape | tipo |
|---|---|---|---|
| entrada | | | |
| saída | | | |

### Pare e responda

- **a)** Qual das quatro dimensões da entrada **não** é um número? O que isso significa na prática?
- **b)** O shape da entrada é `[?, 3, 224, 224]`. O OpenCV devolve `(altura, largura, 3)`.
  As duas coisas batem? Se não, o que precisa acontecer?
- **c)** A saída tem 1000 posições. Já são probabilidades? O arquivo `.onnx` diz isso em algum lugar?

---

## Etapa 2 — Do arquivo JPEG ao tensor 

Arquivo: `02_classificar.py`

A função `preprocessa_ingenuo()` **já vem pronta e você não deve alterá-la**. Ela produz um
tensor com o shape e o dtype exatos que você descobriu na Etapa 1.

**TODO 3** — Chame a função recebida em `funcao_preproc` e confira o tensor contra o contrato.
*Pronto quando:* `tensor.shape == (1, 3, 224, 224)` e `tensor.dtype == float32`.

**TODO 4** — Rode a inferência e extraia o vetor de 1000 números.

- `sessao.run(lista_de_saidas, dicionario_de_entradas)`
- `lista_de_saidas = None` significa "me devolva todas"
- o retorno é uma **lista** de arrays, um por saída
- a saída tem shape `(1, 1000)` — o `1` da frente é o lote

*Pronto quando:* `logits.shape == (1000,)`.

> Neste ponto o programa **roda**. Guarde essa palavra.

---

## Etapa 3 — Do tensor ao nome

Mesmo arquivo.

**TODO 5** — Implemente o `softmax`.

```
softmax(x)_i = exp(x_i) / soma_j exp(x_j)
```

A implementação literal estoura em `float32`. Subtraia o máximo do vetor antes de exponenciar
— o resultado matemático é idêntico.
*Pronto quando:* `softmax([1., 2., 3.])` soma 1.0 **e** `softmax([1000., 1001.])` não dá `nan`.

**TODO 6** — Devolva as `k` maiores como lista de `(rótulo, porcentagem)`, da maior para a
menor. Dica: `np.argsort` ordena crescente; `[::-1]` inverte.

### Rode agora

```bash
python3 02_classificar.py
```

Compare cada resposta com a imagem. **Preencha a coluna INGÊNUO:**

| imagem | o que você vê | top-1 INGÊNUO | conf. | top-1 CORRETO | conf. |
|---|---|---|---|---|---|
| futebol.jpg | | | | | |
| babuino.jpg | | | | | |
| frutas.jpg | | | | | |
| cachorro.jpg | | | | | |

### Perguntas que fecham a etapa

1. O programa deu erro?
2. Olhe a linha `faixa=[min, max]` que o script imprime. O que aquele intervalo diz sobre o
   que você entregou ao modelo?

**TODO 7** — Escreva `preprocessa_correto()`. Compare com o ingênuo e descubra o que falta.
**São três coisas, e nenhuma muda o shape.** Duas você deduz da pergunta 2; a terceira está
na diferença entre como o OpenCV guarda uma imagem e como o modelo foi treinado.

```
média  = [0.485, 0.456, 0.406]
desvio = [0.229, 0.224, 0.225]
```

*Pronto quando:* as 4 imagens são classificadas de forma reconhecível.

**TODO 8** — Descomente a linha do `CORRETO`, rode as 4 imagens com os dois pré-processamentos
e complete a tabela.

> **Guarde a sua `preprocessa_correto()`.** Na segunda metade da aula ela deixa de ser o
> exercício e vira o aparelho de medição.

---

## Etapa 4 — CPU vs CUDA e o aquecimento 

Arquivo: `03_benchmark.py`. Copie para dentro dele a sua `preprocessa_correto()`.

Um *execution provider* é o backend que de fato executa as operações do grafo. O `.onnx` é o
mesmo; o código é o mesmo; quem faz a conta muda.

**TODO 9** — Crie a sessão pedindo um provider e devolva **também** qual o onnxruntime
realmente usou. Duas coisas importam: a lista de providers é de **prioridade**, não uma escolha
única; e **pedir não é obter** — `sessao.get_providers()` conta a verdade.
*Pronto quando:* rodando com `cuda` numa placa sem CUDA, o script avisa que caiu para CPU.

**TODO 10** — Meça a latência e devolva `(primeira_ms, mediana_ms, minimo_ms, maximo_ms)`:
cronometre a **primeira** inferência separadamente; rode `N_AQUECIMENTO` descartando; rode
`N_MEDICOES` guardando; devolva a primeira, a **mediana**, o min e o max.
*Pronto quando:* a coluna `1a (ms)` fica visivelmente maior que a mediana.

| provider | 1ª (ms) | mediana | min | max | top-1 |
|---|---|---|---|---|---|
| CPUExecutionProvider | | | | | |
| CUDAExecutionProvider | | | | | |

*(Se o diagnóstico disse que esta placa não tem CUDA, rode só a linha da CPU — o efeito do
aquecimento, que é o que esta etapa ensina, aparece nela do mesmo jeito.)*

**A conta que fecha o bloco:** calcule a aceleração da GPU duas vezes — pelas medianas, e só
pela primeira inferência. Anote os dois números.

---



# BLOCO B — Quantização

A partir daqui os TODOs ficam quase todos em `motores.py`, importado pelos scripts `04`, `05` e
`06`. Comece copiando para ele a sua `preprocessa_correto()` e o seu `softmax`, sem mudar uma
vírgula.

Um motor é a combinação de **duas escolhas independentes**:

| | modelo FP32 (14 MB) | modelo FP16 (7 MB) |
|---|---|---|
| **CPU EP** | `cpu` | `cpu16` |
| **CUDA EP** | `cuda` | `cuda16` |
| **TensorRT EP** | `trt` | `trt16` — aqui a precisão vem da *flag*, não do arquivo |

Os dois arquivos `.onnx` têm a **mesma interface** (entrada e saída em float32). É por isso que
o seu pipeline do Bloco A serve para os seis sem alteração — e é o que o torna utilizável como
instrumento de medida.

O `00_diagnostico.py` da Etapa 0 já disse quais desses a sua placa consegue rodar.

---

## Etapa 5 — Metade dos bits 

Nada é compilado aqui. A única coisa que muda é **qual arquivo você carrega**.

**Antes de rodar, escreva a sua previsão:** o modelo de 7 MB vai rodar mais rápido, mais lento,
ou igual ao de 14 MB? Por quê?

**TODO 11** — Em `motores.py`, complete `avalia()`. Para cada imagem, devolva `top1`,
`indice1`, `conf1`, `top2`, `conf2` e a **margem** (= probabilidade do 1º − probabilidade do 2º,
em pontos percentuais).

Por que a margem, e não só a confiança? Porque a margem diz o quão **frágil** aquela decisão é.
Uma classificação com 99% de confiança e margem de 0,5 ponto está a um arredondamento de trocar
de resposta.

*Pronto quando:* a `onibus.jpg` aparece com margem de um dígito e as outras quatro com margem de
dois dígitos.

**TODO 12** — Em `motores.py`, complete `cria_sessao()`. Cronometre a criação da
`InferenceSession` e devolva `(sessao, provider_realmente_usado, segundos_de_construcao)`.
Mantenha o `try/except TypeError` — sem ele o script quebra em wheels antigos em vez de degradar.

```bash
python3 04_precisao.py
```

| | FP32 | FP16 |
|---|---|---|
| tamanho do arquivo (MB) | | |
| tempo de sessão (s) | | |
| 1ª inferência (ms) | | |
| mediana (ms) | | |

**A primeira pergunta:** o arquivo encolheu pela metade. E o tempo? Compare com a sua previsão.
Se o resultado te surpreendeu, **a explicação vale mais que o número** — quem faz a conta em
FP16 nesta placa?

**A segunda pergunta**, que você só consegue fazer porque fez o Bloco A:

> **ainda está certo?**

| imagem | top-1 FP32 | margem | top-1 FP16 | margem | trocou? |
|---|---|---|---|---|---|
| futebol.jpg | | | | | |
| babuino.jpg | | | | | |
| frutas.jpg | | | | | |
| cachorro.jpg | | | | | |
| onibus.jpg | | | | | |

**Antes de concluir qualquer coisa:** se nenhuma imagem trocou, isso prova que o FP16 é seguro?
Olhe a coluna de margem — em especial a menor de todas — antes de responder.

---

## Etapa 6 — TensorRT — Desafio

O CUDA EP executa o grafo como ele é, operação por operação. O TensorRT faz outra coisa: lê o
grafo inteiro, funde camadas, testa algoritmos de convolução **na sua placa** para escolher o
mais rápido, e compila um binário específico para aquela GPU. Isso custa caro — e custa **uma
vez**, se você deixar.

**TODO 13** — Em `motores.py`, complete a receita do TensorRT:

```python
"trt_engine_cache_enable": True,
"trt_engine_cache_path":   <uma pasta POR MOTOR>     # PASTA_CACHE_TRT + o motor
"trt_fp16_enable":         True                      # só quando precisao == "fp16"
```

Pasta separada por motor não é estilo, é requisito: FP32 e FP16 geram engines diferentes e uma
sobrescreveria a outra. Se o diagnóstico da Etapa 0 disse que este wheel não aceita
`provider_options`, use as variáveis de ambiente que ele imprimiu.

```bash
python3 05_tensorrt.py trt
```

> **Vai demorar. É para demorar.** Enquanto compila, ninguém mexe na placa.

**TODO 14** — Em `05_tensorrt.py`, imprima o número que justifica o cache: segundos para
**construir** a engine do zero, segundos para **carregar** do cache, a razão entre os dois, e as
duas medianas de inferência lado a lado.

| | segundos |
|---|---|
| construir a engine (cache frio) | |
| carregar do cache (cache quente) | |
| razão | |
| mediana de inferência, frio | |
| mediana de inferência, quente | |

**Pergunte-se:** por que as duas últimas linhas são praticamente iguais, se a primeira coluna
mudou tanto?

---

## Etapa 7 — A matriz 

**TODO 15** — Em `06_matriz.py`, feche a matriz. Para cada motor, numa linha: mediana,
aceleração sobre a referência, **concordância de top-1** com a referência (quantas de quantas
imagens) e o **maior desvio de confiança** em pontos percentuais. Compare **índice**, não o
texto do rótulo.

As duas últimas colunas são o ponto da aula inteira: elas só existem porque o Bloco A construiu
um pipeline que sabe a resposta certa. Sem ele, "2× mais rápido" seria um número sem contrapeso.

```bash
python3 06_matriz.py
```

O script pula sozinho os motores que a sua placa não tem.

| motor | MB | sessão (s) | mediana (ms) | aceleração | concordância | maior desvio |
|---|---|---|---|---|---|---|
| cpu | | | | 1,00× | ref | ref |
| cpu16 | | | | | | |
| cuda | | | | | | |
| cuda16 | | | | | | |
| trt | | | | | | |
| trt16 | | | | | | |

### A decisão (responda por escrito, 5 min)

Qual motor você colocaria em produção para:

- **(a)** um robô que classifica 30 quadros por segundo, ligado o dia todo;
- **(b)** um script disparado por sensor, que roda uma vez e encerra;
- **(c)** um sistema de triagem médica.

Justifique **com números desta tabela**. Os três casos não têm a mesma resposta, e um deles não
se decide por velocidade nenhuma.

---

## Perguntas de análise 

1. O tensor do pré-processamento ingênuo tinha shape certo, dtype certo e número certo de
   canais, e ainda assim a resposta era absurda. **Que categoria de bug é essa, e por que é mais
   perigosa em produção do que um `Exception`?**

2. Uma das imagens continuou sendo classificada corretamente mesmo com o pré-processamento
   ingênuo. Se o seu time tivesse testado só com aquela, teria declarado o sistema pronto.
   **Que prática de teste evita esse desfecho?**

3. Você calculou a aceleração da GPU de duas formas e obteve números muito diferentes.
   **Explique fisicamente o que acontece durante a primeira inferência no CUDA
   ExecutionProvider** e diga em qual cenário de produção esse custo é irrelevante e em qual
   ele é o que mais importa.

4. Trocar de *execution provider* mantendo o FP32 não mudou nenhuma resposta; trocar para o
   modelo FP16 mexeu nas confianças. **Por que a primeira troca é gratuita em resultado e a
   segunda não é?** Responda em termos do que cada uma altera no cálculo.

   E a parte que surpreende: **o modelo FP16 ocupa metade do espaço e pode rodar mais devagar.**
   Explique em que condição isso acontece e o que precisaria ser verdade sobre o hardware para
   que o menor fosse também o mais rápido.

5. Suponha que o seu conjunto de teste tivesse 5000 imagens em vez de 5, e que o motor de menor
   precisão concordasse com a referência em 4993 delas. **Descreva o procedimento que você
   usaria para decidir se esse motor pode ir para produção** — e diga o que você olharia nas 7
   imagens discordantes.

---

## Desafio extraclasse 

Escolha **um**:

**A — Throughput vs. latência.** A dimensão de lote do modelo é dinâmica (TODO 2). Processe as
5 imagens num único tensor `[5, 3, 224, 224]`. Meça latência por lote e imagens por segundo, e
compare com o laço de 5 inferências separadas, nos quatro motores. O ganho de throughput veio
de graça? O que você pagou?

**B — O produtor e o consumidor.** Monte um pipeline de duas threads: uma lendo e
pré-processando imagens, outra inferindo, ligadas por uma fila limitada. Meça o throughput
total e compare com a versão sequencial. Onde está o gargalo — e ele muda quando você troca de
motor?

**C — A engine é portátil?** Copie a pasta `trt_cache/` da sua placa para outra Jetson do
laboratório e rode lá com `--quente`. Funcionou? Se não, leia a mensagem de erro e explique a
que exatamente a engine está amarrada. O que isso implica para distribuir um produto embarcado?

**D — O FP16 na outra ponta.** Se a sua placa tem GPU, rode `python3 04_precisao.py cuda cuda16`
e `python3 04_precisao.py cpu cpu16` e ponha as duas saídas lado a lado. O mesmo par de arquivos
produz razões de latência opostas. Explique por quê, e diga qual das duas medições você
reportaria num datasheet de produto — e por que reportar só uma delas seria desonesto.

Entrega: um `.md` com os números medidos e a resposta.

---
