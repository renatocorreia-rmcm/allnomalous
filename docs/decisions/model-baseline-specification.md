# Especificacao do modelo minimo e do baseline

**Status:** decisao inicial; VRAM e throughput aguardam preflight no Colab.

## Modelo minimo = baseline principal

- **Entrada/dados:** RGB, 512 x 512, resize sem crop e normalizacao ImageNet. Usar os splits fixos existentes (3.262 normais de treino; 367 de validacao), combinando as 15 categorias.
- **Extrator:** `torchvision.models.wide_resnet50_2`, pesos ImageNet V2, congelado em `eval`; usar `layer1`, `layer2` e `layer3`. Alinhar `layer1` (256 canais) e `layer2` (512 canais) por interpolacao bilinear para a grade de `layer3` (1.024 canais, stride 16) e concatenar em um tensor de 1.792 canais. Em entrada 512 x 512, a saida e (B, 1792, 32, 32). `src/feature_extractor.py` implementa esse fluxo usando `torch.no_grad()`; o `forward` recebe somente imagens.
- **Dependencia:** adicionar na implementacao uma versao de torchvision compativel com PyTorch do runtime; ainda nao foi adicionada ao `requirements.txt` nem instalada.
- **Decodificador:** implementado em `src/reconstruction_decoder.py`; projecao 1 x 1 de 1.792 para 128 canais; embedding posicional e 1.024 consultas espaciais aprendidas; Transformer com 2 camadas encoder + 2 decoder, 4 cabecas, feed-forward 512 e dropout 0,1; projecao de volta a 1.792 canais. A loss e MSE media entre features de entrada e reconstruidas.
- **Objetivo/saida:** MSE media para reconstruir features congeladas; mapa por pixel = norma L2 da discrepancia entre features por posicao, bilinear ate 512 x 512; escore de imagem = media dos 1% maiores valores do mapa, implementada em `src/anomaly_scoring.py`.
- **Agregacao escolhida antes do teste:** media do top 1% em vez do maximo; agrega evidencia em varios pixels e reduz sensibilidade a um unico pixel ruidoso. A alternativa maximo apos suavizacao Gaussiana tambem esta implementada, mas nao sera selecionada olhando resultados do teste.
- **Sem:** atencao mascarada de vizinhanca e feature jitter.

## Versao completa

Manter todo o baseline e acrescentar somente (1) mascara de atencao local 7 x 7
nas atencoes do encoder e decoder e (2) jitter de features no treino. O jitter
usa ruido gaussiano escalado pela norma L2 do vetor de canais dividida pelo
numero de canais, com escala 20 e probabilidade 1,0; so altera a entrada do
decoder durante treino, mantendo o alvo original. A mascara restringe cada
posicao a vizinhos espaciais e inclui a propria posicao. A diferenca baseline/completo isola
o efeito conjunto desses dois componentes. Variantes mask-only e jitter-only
ficam opcionais; k-NN nao foi escolhido, pois mudaria a familia do detector.

O teste unitario treina a versao base em um batch sintetico fixo e confirma
reducao da MSE; o decoder completo tambem e testado com forward/backward.
`py -3.9 scripts/smoke_test_decoder.py --category bottle` executa um forward
do extrator e decoder completo em uma imagem normal real da categoria indicada.
O checkpoint de pipeline em `notebooks/category_pipeline_checkpoint.ipynb`
tambem executa DataLoader -> extrator -> decoder -> mapa -> escore em uma
amostra defeituosa de `bottle`; o decoder nessa verificacao ainda nao foi
treinado, portanto o resultado confirma somente integracao e alinhamento.

Antes do treino completo, `scripts/quick_category_experiment.py` treina o
baseline em uma categoria por poucas epocas, mede loss de treino/validacao e
cria um defeito sintetico somente a partir de imagens normais de validacao.
O protocolo deixa o teste oficial fora do ciclo de selecao. A rodada curta
documenta resolucao, batch, tempo por epoca e limites do hardware, sem alterar
os hiperparametros previamente definidos a partir de resultados de teste.
O primeiro resultado (duas epocas de `bottle`, CPU, 256 x 256) esta registrado
em `docs/experiments/quick-validation-bottle.md`; o tempo extrapolado nao se
aplica a GPU nem a entrada 512 x 512.

## Hiperparametros e comparacao

| Seed | Batch/GPU (acum.) | Otimizador | Epocas |
|---|---:|---|---:|
| 42 | 2 (2; efetivo 4) | AdamW, lr 1e-4, betas (0,9; 0,999), weight decay 1e-4 | 30 max.; cosine scheduler; early stopping 5 epocas pela loss normal de validacao |

Baseline e completo usam mesmos pesos, splits, seed, resize/normalizacao, batch,
loss, otimizador, scheduler e criterio de parada. Categorias servem somente
para agrupar metricas; labels e mascaras do teste sao exclusivos da avaliacao.
Executar o teste oficial uma vez e comparar AUROC de imagem, AUROC de pixel e
AUPRO por categoria e macro-media.

## GPU e estimativa de treino

O ambiente disponivel tem PyTorch 2.8.0 CPU-only e torchvision 0.23.0
(`cuda.is_available() == False`); **nenhum limite de VRAM/tempo de GPU foi
medido**. Um batch 512 x 512 com pesos ImageNet V2 produziu a saida esperada;
em CPU, a latencia media aquecida foi 722,6 ms/batch (10 iteracoes, batch 1).
Isso nao estima desempenho em GPU. O script
`py -3.9 scripts/benchmark_feature_extractor.py` informa latencia e, quando
CUDA esta disponivel, VRAM maxima alocada/reservada. Batch 2 e acumulacao 2
continuam provisorios. No Colab, medir VRAM e tempo de um epoch; em caso de
falta de memoria, usar batch 1/acumulacao 4, preservando a entrada 512 x 512.

Com 3.262 imagens, sao 1.631 batches/epoca e 48.930 batches por modelo em 30
epocas. Estimativa de planejamento (nao medida): 0,25-1 s/batch da 3,4-13,6 h
por modelo, ou 6,8-27,2 h para os dois, sem validacao. Recalcular com o
throughput medido antes de fixar o numero final de epocas.

**Atribuicao:** arquitetura propria, inspirada mas nao equivalente a UniAD;
nenhum codigo UniAD/anomalib copiado. Citar no notebook e relatorio PyTorch,
torchvision/Wide-ResNet-50-2, Wide Residual Networks, ImageNet, NumPy, Pillow,
artigo e [repositorio oficial UniAD (Apache-2.0)](https://github.com/zhiyuanyou/UniAD),
registrando versoes. Verificar a licenca dos pesos ImageNet antes de redistribuir
checkpoints.
