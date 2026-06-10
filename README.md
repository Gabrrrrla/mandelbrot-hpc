## Mandelbrot MPI

Esse trabalho foi realizado pelas alunas Ana Beatriz Stahl, Emanuele Thomazzoni, Gabriela Rodrigues e Luisa Becker para a Atividade Acadêmica de Computação de Alto Desempenho ministrada pelo Prof. Dr. Rodrigo da Rosa Righi.

O obtivo é fazer a geração paralela do conjunto de Mandelbrot em C com MPI, utilizando esquema mestre-trabalhador com balanceamento dinâmico de carga. A cada tile computado, um frame parcial PPM é salvo, permitindo visualizar a construção da imagem ao longo do tempo e evidenciar o paralelismo em ação.

---

## Sumário

1. [Sobre o projeto](#1-sobre-o-projeto)
2. [Arquitetura](#2-arquitetura)
3. [O conjunto de Mandelbrot](#3-o-conjunto-de-mandelbrot)
4. [Paralelismo com MPI](#4-paralelismo-com-mpi)
5. [Parâmetros](#5-parâmetros)
6. [Compilar e executar](#6-compilar-e-executar)
7. [Saída](#7-saída)
8. [Experimentos sugeridos](#8-experimentos-sugeridos)
9. [Converter frames para animação](#9-converter-frames-para-animação)
10. [Dependências](#10-dependências)

---

## 1. Sobre o projeto

Este projeto implementa a geração do conjunto de Mandelbrot de forma paralela usando MPI (*Message Passing Interface*). O trabalho de computação é distribuído entre múltiplos processos, reduzindo o tempo total de geração da imagem.

Além do resultado final, o programa exporta **frames intermediários** que mostram visualmente como os blocos são preenchidos em ordem não-sequencial, refletindo a natureza assíncrona do processamento paralelo.

---

## 2. Arquitetura

O programa segue o padrão **mestre-trabalhador** (*master-worker*).

```
┌─────────────────────────────────────────────────┐
│              Processo Mestre (rank 0)            │
│                                                  │
│  - Divide a imagem em tiles                      │
│  - Distribui tiles para workers disponíveis      │
│  - Recebe tiles computados                       │
│  - Monta a imagem acumulada                      │
│  - Salva frame PPM a cada tile recebido          │
│  - Envia TAG_STOP quando não há mais trabalho    │
└────────────────────┬────────────────────────────┘
                     │ MPI Send/Recv
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Worker 1 │ │ Worker 2 │ │ Worker N │
  │          │ │          │ │          │
  │ Computa  │ │ Computa  │ │ Computa  │
  │ pixels   │ │ pixels   │ │ pixels   │
  │ do tile  │ │ do tile  │ │ do tile  │
  └──────────┘ └──────────┘ └──────────┘
```

### Tags MPI

| Tag | Valor | Descrição |
|---|---|---|
| `TAG_WORK` | 1 | Mestre envia tile para worker processar |
| `TAG_DONE` | 2 | Worker devolve resultado ao mestre |
| `TAG_STOP` | 3 | Mestre sinaliza encerramento ao worker |

### Fluxo de execução

1. Mestre divide a imagem em tiles de `square_size × square_size` pixels
2. Mestre envia um tile inicial para cada worker disponível
3. Worker computa os pixels e devolve com `TAG_DONE`
4. Mestre copia o tile na imagem, salva o frame e envia o próximo tile disponível
5. Quando todos os tiles foram processados, mestre envia `TAG_STOP` para cada worker
6. Mestre grava a imagem final completa em disco

### Balanceamento dinâmico de carga

Ao contrário de uma divisão estática, o balanceamento é **dinâmico**: um worker que termina seu tile mais rápido recebe imediatamente o próximo disponível. Isso é especialmente importante no Mandelbrot, onde regiões diferentes têm custos computacionais muito distintos — pontos no interior do conjunto atingem `max_iter` iterações, enquanto pontos externos divergem rapidamente.

---

## 3. O conjunto de Mandelbrot

O conjunto de Mandelbrot é definido no plano complexo. Para cada ponto `c = cx + i·cy`, itera-se a função:

```
z₀ = 0
z_{n+1} = z²_n + c
```

Se `|z|` não diverge após `max_iter` iterações, o ponto pertence ao conjunto (pintado de preto). Caso contrário, a cor é determinada pelo número de iterações até a divergência.

### Mapeamento de pixels para o plano complexo

```c
double cx = ((double)x / hxres - 0.5) / magnify * 3.0 - 0.7;
double cy = ((double)y / hyres - 0.5) / magnify * 3.0;
```

- O offset `-0.7` em `cx` centraliza a vista na região mais interessante do conjunto
- `magnify` controla o zoom: valores maiores aproximam a imagem
- A condição de divergência usada é `|z|² > 4`, equivalente a `|z| > 2`

### Colorização

```c
R = 0
G = (it * 2)  % 50
B = (it * 12) % 150
```

Pixels que escapam do conjunto recebem um gradiente azul-esverdeado proporcional à velocidade de divergência. Pontos no interior são pintados de preto.

---

## 4. Paralelismo com MPI

### Por que MPI?

MPI é uma interface de troca de mensagens para programação paralela em memória distribuída. Diferente de threads, processos MPI se comunicam por mensagens explícitas, sendo adequado tanto para máquinas multicore quanto para clusters.

### Por que tiles?

Dividir a imagem em tiles quadrados permite granularidade configurável. Tiles menores geram mais tarefas e melhor balanceamento, ao custo de maior overhead de comunicação. Tiles maiores reduzem o overhead, mas podem causar desequilíbrio se uma região for muito mais custosa que outra.

Nos frames intermediários, é visível que os blocos aparecem em posições aleatórias — mostrando que diferentes workers processam regiões distintas simultaneamente.

### Overhead de comunicação

Cada tile gera duas trocas de mensagens:

- **Mestre → Worker:** metadados do tile (`Tile`, 16 bytes)
- **Worker → Mestre:** metadados (`Tile`, 16 bytes) + pixels (`w × h × 3` bytes)

Para um tile de `50×50`, isso representa `7.500 bytes` de dados. O overhead é baixo comparado ao custo de computação, especialmente com `max_iter` alto.

---

## 5. Parâmetros

```
mpirun -np <processos> ./mandelbrot <square_size> <max_iter> <magnify>
```

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `processos` | inteiro | Total de processos MPI (1 mestre + N workers) |
| `square_size` | inteiro | Lado de cada tile em pixels |
| `max_iter` | inteiro | Número máximo de iterações por pixel (complexidade) |
| `magnify` | float | Fator de zoom sobre o conjunto |

### Impacto de cada parâmetro

| Parâmetro | Valor baixo | Valor alto |
|---|---|---|
| `processos` | Menos paralelismo, mais lento | Mais paralelismo, mais rápido |
| `square_size` | Mais tiles, mais frames, mais overhead | Menos tiles, menos frames, menos overhead |
| `max_iter` | Imagem menos detalhada, mais rápido | Imagem mais detalhada, mais lento |
| `magnify` | Visão geral do conjunto | Zoom em regiões de alta complexidade |

---

## 6. Compilar e executar

### Instalar dependências

**Ubuntu/Debian**
```bash
sudo apt install libopenmpi-dev openmpi-bin
```

**Fedora/RHEL**
```bash
sudo dnf install openmpi openmpi-devel
```

**macOS (Homebrew)**
```bash
brew install open-mpi
```

**Windows**

Recomendado usar o [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi):

1. Baixe e instale o `msmpisetup.exe` e o `msmpisdk.msi` na página oficial
2. Instale o [MinGW-w64](https://www.mingw-w64.org/) ou use o compilador do Visual Studio
3. Compile com:
```bash
mpicc -O2 -o mandelbrot mandelbrot.c -lm
```

Alternativamente, use o **WSL** (Windows Subsystem for Linux) e siga as instruções do Ubuntu.

### Compilar

```bash
mpicc -O2 -o mandelbrot mandelbrot.c -lm
```

### Executar

```bash
mpirun -np <processos> ./mandelbrot <square_size> <max_iter> <magnify>
```

### Exemplo básico

```bash
mpirun -np 4 ./mandelbrot 50 1000 1.0
```

---

## 7. Saída

| Arquivo | Descrição |
|---|---|
| `frame_0000.ppm` ... `frame_XXXX.ppm` | Frames parciais, um por tile recebido |
| `mandelbrot_final.ppm` | Imagem completa ao término da execução |

O número total de frames é igual ao número de tiles:

```
frames = ceil(hxres / square_size) × ceil(hyres / square_size)
```

Para `hxres = hyres = 500` e `square_size = 50`, são gerados `10 × 10 = 100` frames.

O formato PPM (*Portable Pixmap*) é sem compressão e pode ser aberto no GIMP, feh, ou convertido com ImageMagick e ffmpeg.

---

## 8. Experimentos sugeridos

### Variando o número de processos

```bash
time mpirun -np 2 ./mandelbrot 50 2000 1.0
time mpirun -np 4 ./mandelbrot 50 2000 1.0
time mpirun -np 8 ./mandelbrot 50 2000 1.0
```

### Variando o tamanho do tile

```bash
# Tiles pequenos: muitos frames, balanceamento fino
mpirun -np 4 ./mandelbrot 10 1000 1.0

# Tiles médios
mpirun -np 4 ./mandelbrot 50 1000 1.0

# Tiles grandes: poucos frames, menos overhead
mpirun -np 4 ./mandelbrot 100 1000 1.0
```

### Variando a complexidade

```bash
# Baixa complexidade
mpirun -np 4 ./mandelbrot 50 100 1.0

# Alta complexidade
mpirun -np 4 ./mandelbrot 50 5000 1.0
```

### Zoom profundo

```bash
mpirun -np 8 ./mandelbrot 25 8000 10.0
```

---

## 9. Converter frames para animação

### GIF animado (ImageMagick)

```bash
convert -delay 5 -loop 0 frame_*.ppm mandelbrot.gif
```

### Vídeo MP4 (ffmpeg)

```bash
ffmpeg -framerate 10 -i frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p mandelbrot.mp4
```

### Visualizar PPM diretamente

```bash
# feh
feh mandelbrot_final.ppm

# Converter para PNG
convert mandelbrot_final.ppm mandelbrot_final.png
```

---

## 10. Dependências

| Dependência | Uso | Obrigatória |
|---|---|---|
| OpenMPI ou MPICH | Execução paralela MPI | Sim |
| `mpicc` | Compilação | Sim |
| ImageMagick | Conversão PPM → GIF/PNG | Não |
| ffmpeg | Conversão PPM → MP4 | Não |
| feh / GIMP | Visualização PPM | Não |
