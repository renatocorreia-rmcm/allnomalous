# allnomalous

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
