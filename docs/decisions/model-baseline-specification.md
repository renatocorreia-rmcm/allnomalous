# Especificacao do modelo minimo e do baseline

**Status:** decisao inicial; VRAM e throughput aguardam preflight no Colab.

## Modelo minimo = baseline principal

- **Entrada/dados:** RGB, 512 x 512, resize sem crop e normalizacao ImageNet. Usar os splits fixos existentes (3.262 normais de treino; 367 de validacao), combinando as 15 categorias.
- **Extrator:** `torchvision.models.wide_resnet50_2`, pesos ImageNet V2, congelado em `eval`; usar `layer3` (stride 16, 1.024 canais, grid 32 x 32). O `forward` recebe somente imagens.
- **Dependencia:** adicionar na implementacao uma versao de torchvision compativel com PyTorch do runtime; ainda nao foi adicionada ao `requirements.txt` nem instalada.
- **Decodificador:** projecao 1 x 1 de 1.024 para 128 canais; 1.024 consultas espaciais aprendidas; Transformer com 2 camadas encoder + 2 decoder, 4 cabecas, feed-forward 512 e dropout 0,1; projecao de volta a 1.024 canais.
- **Objetivo/saida:** MSE media para reconstruir features congeladas; mapa por pixel = RMSE dos canais por posicao, bilinear ate 512 x 512; escore de imagem = media do 1% de pixels com maior escore.
- **Sem:** atencao mascarada de vizinhanca e feature jitter.

## Versao completa

Manter todo o baseline e acrescentar somente (1) mascara de atencao local 7 x 7
no Transformer e (2) jitter de features. Ponto de partida de jitter da
configuracao oficial UniAD: escala 20, probabilidade 1,0; validar a intensidade
em um smoke test antes do treino completo. A diferenca baseline/completo isola
o efeito conjunto desses dois componentes. Variantes mask-only e jitter-only
ficam opcionais; k-NN nao foi escolhido, pois mudaria a familia do detector.

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

O ambiente disponivel tem PyTorch 2.8.0 CPU-only (`cuda.is_available() == False`);
**nenhum limite de VRAM/tempo de GPU foi medido**. Batch 2 e acumulacao 2 sao
provisorios. No Colab, medir VRAM e tempo de um epoch; em caso de falta de
memoria, usar batch 1/acumulacao 4, preservando a entrada 512 x 512.

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
