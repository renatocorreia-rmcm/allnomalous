# allnomalous

## Avaliação final no MVTec AD

A avaliação oficial usou 1.725 imagens das 15 categorias. O baseline era da época 109 e o modelo completo da época 315.

| Modelo | AUROC imagem | AUROC pixel | AUPRO |
|---|---:|---:|---:|
| Baseline | 78,40% | 89,08% | 78,56% |
| Completo | 78,46% | 89,08% | 78,75% |

- [Índice da avaliação, checkpoints e instruções de reanálise](analysis/final_eval_315/README.md)
- [Tabela por categoria em CSV](analysis/final_eval_315/final_metrics.csv) e [LaTeX](analysis/final_eval_315/tabela_final.tex)
- [Relatório de avaliação e análise de falhas](analysis/final_eval_315/relatorio_avaliacao.md)
- [Mapas de objetos](analysis/final_eval_315/casos_objetos.png), [de texturas](analysis/final_eval_315/casos_texturas.png) e [distribuição dos escores normais](analysis/final_eval_315/escores_normais.png)

O [avaliador congelado](https://github.com/renatocorreia-rmcm/allnomalous/commit/34f28c571d55291a12fc06d63aacdb4dac503abd) e o [manifesto da rodada](analysis/final_eval_315/final_test_run.json) identificam o código e os checkpoints usados. Os diferentes orçamentos e critérios de seleção limitam a atribuição do pequeno ganho de AUPRO à máscara e ao jitter.

Class independent anomaly detection model

## MVTec AD

This repository includes the MVTec AD dataset. Images are stored with Git LFS;
install Git LFS before cloning, then run `git lfs pull` to download them.
The dataset is provided under the terms in [license.txt](license.txt).
Please cite Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten
Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly
Detection," CVPR 2019.

## Divisao treino/validacao

### Criar a divisao

Execute `python create_splits.py` na raiz desta base. O script usa semente
`42` e reserva 10% das imagens normais de treino em cada categoria para
validacao. A quantidade reservada e arredondada para cima por categoria; o
restante fica no treino.

O script gera `train_split.csv` e `val_split.csv`, ambos com as colunas
`category` e `path`. Os caminhos sao relativos a raiz da base. A divisao
considera somente imagens em `<categoria>/train/good`; portanto, a validacao
contem apenas imagens normais. Os diretorios de teste oficial e de
`ground_truth` nao sao lidos nem modificados.

Para recriar a mesma divisao, mantenha a semente `42` e a mesma versao dos
arquivos da base. A funcao `create_splits` em `create_splits.py` tambem aceita
uma semente e uma fracao de validacao explicitas.

Use a validacao para acompanhar a perda de reconstrucao e ajustar
hiperparametros. Ela nao deve ser usada para medir AUROC; essa avaliacao deve
usar o teste oficial.

## Figuras de exemplo do dataset

Gere as figuras de exemplo em PNG com resolucao de 300 dpi executando na raiz
do repositorio:

```bash
python plot_dataset_examples.py
```

No Windows, tambem e possivel usar `py -3.9 plot_dataset_examples.py`. Os
arquivos sao salvos em `figures/dataset_examples/`. Para alterar a pasta dos
dados ou a pasta de saida, use `--data-root` e `--output-dir`; `--dpi` altera a
resolucao (o padrao e 300).

A funcao `plot_example(data_root, category, defect_type, image_name=None)`
em `plot_dataset_examples.py` monta tres paineis: uma imagem normal de treino,
uma imagem de teste defeituosa e a mesma imagem com a mascara ground truth em
vermelho. As figuras geradas incluem as legendas:

1. **Categorias normais:** uma imagem normal de treino (`good`) para cada uma
   das 15 categorias do MVTec AD.
2. **Bottle — broken large:** imagem normal de treino, exemplo de teste com
   quebra grande e mascara ground truth sobreposta.
3. **Carpet — cut:** imagem normal de treino, exemplo de teste com corte e
   mascara ground truth sobreposta.
4. **Transistor — damaged case:** imagem normal de treino, exemplo de teste
   com dano na carcaça e mascara ground truth sobreposta.

## Resolucao e tamanho dos defeitos

Execute `py -3.9 analyze_dataset.py` na raiz do repositorio para recriar as
tabelas e o histograma em `analysis/dataset_profile/`. A leitura das imagens
usa apenas os metadados de tamanho e modo do PIL; as mascaras sao lidas uma de
cada vez para contar pixels marcados (`> 0`). A proporcao e calculada como
pixels marcados divididos pela area da imagem correspondente.

- `image_profiles_by_category.csv`: resolucao em altura x largura, modo de cor
  e quantidade de imagens por perfil.
- `mask_anomaly_ratios.csv`: dimensoes da imagem e mascara, contagem de pixels
  anômalos, proporcao por mascara e verificacao de tamanho.
- `anomaly_summary_by_category.csv`: quantidade, media, mediana, menor
  proporcao e contagem de mascaras pequenas por categoria.
- `anomaly_ratio_histogram.png`: distribuicao das proporcoes; a linha em 1% e
  uma referencia visual para destacar defeitos pequenos.

Na versao analisada, as 5.354 imagens tem tamanhos entre 700 x 700 e
1024 x 1024. As categorias `grid`, `screw` e `zipper` estao no modo `L`
(tons de cinza); as outras 12 estao em `RGB`. Foram analisadas 1.258 mascaras
e todas tem o mesmo tamanho da imagem correspondente. A proporcao media e a
mediana de cada categoria estao na tabela de resumo.

A menor mascara encontrada e `capsule/crack/021_mask.png`: 371 pixels em uma
imagem de 1000 x 1000 (0,0371%). No conjunto, 499 mascaras ocupam menos de 1%
da imagem e 17 ocupam menos de 0,1%. Como estimativa pela area, a menor mascara
teria cerca de 97 pixels marcados em 512 x 512, contra 24 em 256 x 256 e 6 em
128 x 128. Isso nao garante que a forma do defeito sobreviva ao redimensionamento:
interpolacao e defeitos finos podem apagar ou fragmentar pixels.

Como ponto de partida, recomenda-se 512 x 512, comparando o desempenho com a
resolucao nativa para as categorias e mascaras menores antes de fixar a entrada
do modelo. Evite reduzir diretamente para 128 x 128; a avaliacao deve acompanhar
recall nas amostras com menor proporcao de mascara.

## DataLoader PyTorch

Instale as dependencias com `py -3.9 -m pip install -r requirements.txt`.
Os datasets e as funcoes para criar os loaders estao em
`src/mvtec_dataset.py`. O tamanho padrao e 512 x 512; imagens sao convertidas
para RGB, redimensionadas sem recorte e normalizadas com a media e o desvio
padrao ImageNet.

```python
from src.mvtec_dataset import create_dataloaders, create_test_dataloader

train_loader, validation_loader = create_dataloaders(".", batch_size=32)
test_loader = create_test_dataloader(".", batch_size=32)

for images in train_loader:
    output = model(images)  # O modelo recebe apenas as imagens.

for images, masks, labels, categories in test_loader:
    output = model(images)  # categories sao metadados para agrupar a avaliacao.
```

O treino e a validacao leem `train_split.csv` e `val_split.csv`, respectivamente;
ambos combinam as 15 categorias e retornam somente tensores de imagem, sem
rótulo nem metadado de categoria. O loader de treino embaralha as amostras. O
loader de teste retorna imagem, mascara binaria, rotulo de imagem (0 normal,
1 defeituosa) e nome da categoria. As imagens e mascaras usam o mesmo tamanho;
mascaras sao redimensionadas com interpolacao nearest e imagens com interpolacao
bilinear. Imagens normais de teste recebem uma mascara zerada.

Para validar a implementacao, execute:

```bash
py -3.9 -m unittest discover -s tests
py -3.9 -c "from src.mvtec_dataset import create_dataloaders; train, val = create_dataloaders('.', batch_size=8); print('train', next(iter(train)).shape, 'validation', next(iter(val)).shape)"
```

## Decisao de implementacao do modelo

A comparacao entre a implementacao oficial UniAD e anomalib, a decisao
arquitetural e o cronograma revisado estao em
[`docs/decisions/uniad-implementation.md`](docs/decisions/uniad-implementation.md).
A especificacao detalhada do modelo minimo e completo, do baseline de ablation
e do protocolo de comparacao esta em
[`docs/decisions/model-baseline-specification.md`](docs/decisions/model-baseline-specification.md).
O relatorio tambem registra a decisao e as citacoes de artigos, bibliotecas e
implementacoes consultadas.

## Diagramas e pseudocodigos

Os diagramas SVG do pipeline do modelo e do protocolo experimental, junto com
os pseudocodigos de treino e inferencia, estao em
[`docs/model-pipeline-pseudocode.md`](docs/model-pipeline-pseudocode.md).

## Checkpoint do pipeline em uma categoria

O notebook [`notebooks/category_pipeline_checkpoint.ipynb`](notebooks/category_pipeline_checkpoint.ipynb)
executa DataLoader, extrator, decoder, mapa de discrepancia e escore em uma
amostra defeituosa de `bottle`. O decoder ainda nao foi treinado; esse
checkpoint verifica execucao e alinhamento, nao qualidade do detector. A
figura gerada fica em `figures/category_pipeline_bottle.png`, e o cronograma
revisado esta em
[`docs/decisions/uniad-implementation.md`](docs/decisions/uniad-implementation.md).

## Treinamento combinado no Google Colab

O notebook [`notebooks/train_combined_colab.ipynb`](notebooks/train_combined_colab.ipynb)
clona este repositorio publico e baixa as imagens MVTec armazenadas em Git LFS;
assim, o codigo e os dados vem do proprio GitHub, sem upload manual do dataset.
Ele monta o Google Drive para salvar checkpoint retomavel a cada epoca, melhor
modelo, historico e curva de loss em `MyDrive/allnomalous_training`. O notebook
tambem oferece uma busca curta de tres configuracoes em subconjuntos normais de
treino/validacao, salva `hyperparameter_search.csv` e mapas para inspecao,
orienta a testar a retomada apos reiniciar o runtime e exporta
`computational_costs.csv` com parametros, tamanho dos checkpoints, tempos,
latencia de inferencia e memoria da GPU. Abra o notebook no Colab, selecione
uma GPU, execute as celulas em ordem, escolha e justifique a configuracao apos
inspecionar os mapas e rode `train_block()` novamente para continuar por blocos.

## Experimento curto antes do treino completo

Execute `py -3.9 scripts/quick_category_experiment.py --category bottle
--image-size 256 --epochs 2 --batch-size 1` para treinar o baseline em imagens
normais de treino da categoria e acompanhar a loss em imagens normais de
validacao. O mesmo script cola um recorte de outra imagem normal de validacao
para criar uma anomalia sintetica; nenhuma imagem do teste oficial e usada.
Curva, comparacao visual, mapas brutos e `metrics.json` sao salvos em
`analysis/quick_validation_bottle/`. O teste em resolucao reduzida e CPU serve
como verificacao de aprendizado e depuracao, nao como ajuste final de
hiperparametros ou estimativa de desempenho em GPU.
Resultados, interpretacao e limitacoes estao em
[`docs/experiments/quick-validation-bottle.md`](docs/experiments/quick-validation-bottle.md).

## Extrator de caracteristicas

O extrator congelado Wide-ResNet-50-2 que alinha e concatena as features de
`layer1`, `layer2` e `layer3` esta em `src/feature_extractor.py`. Para entrada
512 x 512, sua saida tem forma `(B, 1792, 32, 32)`. Rode
`py -3.9 scripts/benchmark_feature_extractor.py` para medir latencia e, quando
CUDA estiver disponivel, memoria de pico. Os pesos ImageNet V2 sao baixados na
primeira execucao. No ambiente CPU-only usado nesta validacao, o batch 1 levou
722,6 ms em media apos aquecimento; esse resultado nao estima a velocidade em
GPU.

## Decodificador de reconstrucao

`src/reconstruction_decoder.py` implementa o decoder com embeddings espaciais
e consultas aprendidas. A configuracao baseline desativa jitter e mascara;
configure `jitter_scale=20`, `jitter_probability=1` e `neighbor_size=7` para a
versao completa. Jitter so e aplicado em modo treino, e a mascara restringe a
atencao a vizinhanca local. A loss MSE e calculada com `reconstruction_loss`.
Valide com `py -3.9 -m unittest discover -s tests`; para testar uma amostra
normal real, execute `py -3.9 scripts/smoke_test_decoder.py --category bottle`.

## Escore por imagem

`src/anomaly_scoring.py` agrega mapas de um batch em escores de imagem. O
metodo predefinido `topk_mean` calcula a media dos 1% maiores pixels, escolhida
para reduzir sensibilidade a um unico pico ruidoso. A alternativa `max` aplica
suavizacao Gaussiana 5 x 5 antes do maximo. Ambos aceitam mapas `(B,H,W)` ou
`(B,1,H,W)` e retornam `(B,)`; testes de batch estao em
`tests/test_anomaly_scoring.py`.

## Mapas de anomalia por pixel

`src/anomaly_maps.py` calcula a norma L2 da discrepancia por posicao, redimensiona
bilinearmente e retorna mapas brutos `float32` `(B,H,W)`. Suavizacao Gaussiana
e opcional. A normalizacao min-max esta isolada em
`normalize_map_for_visualization` e nao deve ser usada nas metricas. Gere uma
previa alinhada a imagem e mascara com
`py -3.9 scripts/visualize_anomaly_maps.py`; a figura resultante identifica o
decoder como nao treinado e serve apenas para validar o pipeline espacial.

## Metricas de deteccao

`src/anomaly_metrics.py` expoe `evaluate_model(extractor, decoder, loader)`,
que calcula AUROC por imagem, AUROC por pixel e AUPRO ate FPR 0,3 por
categoria e media macro. O AUROC por pixel e o AUPRO usam histogramas
configuraveis (2048 bins por padrao), evitando guardar todos os mapas na
memoria; o AUROC por imagem usa ordenacao exata. AUPRO usa componentes
conectados 4-vizinhos e o pacote SciPy declarado em `requirements.txt`.
Com o limite FPR de 0,3, uma curva aleatoria diagonal tem AUPRO esperado em
torno de 0,15; AUROC aleatorio fica em torno de 0,5. Os testes usam apenas
mapas sinteticos e nao acessam o conjunto oficial de teste.

### Contagens por categoria

| Categoria | Normais originais | Treino | Validacao |
|---|---:|---:|---:|
| bottle | 209 | 188 | 21 |
| cable | 224 | 201 | 23 |
| capsule | 219 | 197 | 22 |
| carpet | 280 | 252 | 28 |
| grid | 264 | 237 | 27 |
| hazelnut | 391 | 351 | 40 |
| leather | 245 | 220 | 25 |
| metal_nut | 220 | 198 | 22 |
| pill | 267 | 240 | 27 |
| screw | 320 | 288 | 32 |
| tile | 230 | 207 | 23 |
| toothbrush | 60 | 54 | 6 |
| transistor | 213 | 191 | 22 |
| wood | 247 | 222 | 25 |
| zipper | 240 | 216 | 24 |
| **TOTAL** | **3629** | **3262** | **367** |
