# Experimento curto: baseline em bottle

**Decisao:** o baseline aprendeu a reduzir a loss em duas epocas e o pipeline
reagiu a uma alteracao sintetica; seguir para o preflight GPU antes do treino
completo. O teste oficial nao foi lido nem usado.

## Protocolo

- Categoria: `bottle`.
- Treino: 188 imagens normais de `train/good`; validacao: 21 imagens normais
  do split fixo. Apenas o decoder foi treinado; extrator ImageNet V2 congelado.
- Baseline: sem jitter e sem mascara de atencao; AdamW, `lr=1e-4`, batch 1,
  seed 42, duas epocas.
- Resolucao: 256 x 256, usada apenas para manter o teste executavel no host
  CPU com memoria limitada. A configuracao de estudo continua candidata a
  512 x 512.
- Checagem sintetica: colagem de um recorte de outra imagem normal de validacao
  sobre uma imagem normal `bottle`. Nenhuma imagem, label ou mascara do teste
  oficial participou do treino, da validacao ou da checagem.

## Resultados

| Epoca | MSE treino | MSE validacao normal | Tempo (CPU) |
|---:|---:|---:|---:|
| 1 | 0.985617 | 0.602882 | 66.9 s |
| 2 | 0.542846 | 0.466915 | 63.2 s |

A MSE caiu 44.9% no treino e 22.6% na validacao entre a primeira e a segunda
epoca. O tempo medio medido foi 65.1 s/epoca para 188 amostras de treino e 21
de validacao, em CPU, batch 1, 256 x 256.

A media do top 1% dos pixels foi 43.43 para uma imagem normal da validacao e
60.07 para a imagem com colagem. Na imagem sintetica, a discrepancia media
dentro do patch foi 50.22, contra 29.78 fora dele (razao 1.69). E um sinal
positivo para a verificacao sintetica, nao uma metrica de desempenho do modelo.

## Artefatos

Todos estao em `analysis/quick_validation_bottle/`:

- `loss_curve.png`: curva de MSE de treino e validacao.
- `validation_and_synthetic_examples.png`: imagem normal e mapa, imagem com
  patch artificial, mascara sintetica e mapa sobreposto.
- `losses.csv` e `metrics.json`: perdas, tempos e resumo das medidas.
- `normal_validation_map_raw.npy` e `synthetic_defect_map_raw.npy`: mapas
  float32 sem normalizacao para visualizacao.

![Curva de loss do teste curto](../../analysis/quick_validation_bottle/loss_curve.png)

![Validacao normal e alteracao sintetica](../../analysis/quick_validation_bottle/validation_and_synthetic_examples.png)

## Limites e estimativa

Uma extrapolacao linear ingenua do tempo de epoca para 3.262 imagens e 30
epocas resulta em aproximadamente 9.4 horas por modelo (18.8 horas para
baseline e completo), mantendo **CPU, 256 x 256 e batch 1**. E somente uma
referencia de escala: o custo de validacao e overhead nao escalam exatamente
com o treino, e o teste nao mediu a configuracao de 512 x 512.

Nao foi medida GPU/VRAM; portanto, nao ha estimativa valida de tempo em GPU nem
do treino na resolucao candidata de 512 x 512. Fazer o preflight nessa
resolucao e dispositivo antes de agendar o treino completo. Duas epocas de
uma categoria nao estabelecem generalizacao para as 15 categorias. Os mapas
apresentados usam somente validacao normal e perturbacao artificial, sem olhar
o teste.
