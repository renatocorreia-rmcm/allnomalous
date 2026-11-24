# Diagramas e pseudocodigos do pipeline

Os diagramas abaixo estao em SVG vetorial e podem ser inseridos diretamente no
relatorio ou slides.

- [Pipeline do modelo](figures/model-pipeline.svg)
- [Protocolo experimental](figures/experimental-protocol.svg)

O extrator implementado em `src/feature_extractor.py` congela
`torchvision.models.wide_resnet50_2` com pesos ImageNet V2, mantem o backbone em
`eval` e usa `torch.no_grad()`. `layer1` (256 canais) e `layer2` (512 canais)
sao redimensionadas por interpolacao bilinear a grade de `layer3` (1024 canais)
e concatenadas. Com imagens 512 x 512, a forma resultante e
`(batch, 1792, 32, 32)`.

## Pseudocodigo de treino

Treinar baseline e versao completa separadamente, usando os mesmos splits,
seed, pesos do extrator, transforms e hiperparametros. O extrator permanece
congelado; somente o decoder e otimizado.

```text
funcao TREINAR(versao, train_loader, val_loader):
    fixar_seed(42)
    extrator <- WideResNet50_2(pesos=ImageNet_V2).eval().congelar()
    decoder <- Decoder(
        projecao=1792->128,
        embedding_posicional=aprendido,
        consultas=1024,
        camadas_encoder=2,
        camadas_decoder=2,
        cabecas=4,
        feedforward=512,
        neighbor_size=7 se versao == COMPLETA; senao nenhuma,
        jitter_scale=20 e jitter_probability=1.0 se versao == COMPLETA;
        jitter_scale=0 e jitter_probability=0 se versao == BASELINE
    )
    otimizador <- AdamW(decoder, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler <- CosineAnnealingLR(T_max=30)
    melhor_loss <- infinito
    epocas_sem_melhora <- 0

    para epoca em 1..30:
        decoder.train()
        para passo, imagens em enumerate(train_loader):  # apenas imagens; 15 classes misturadas
            alvo <- extrator(imagens).detach()  # layer1-3 alinhadas; sem gradiente
            # O decoder aplica jitter so em treino; alvo permanece intacto.
            reconstruido <- decoder(alvo)
            loss <- MSE(reconstruido, alvo)
            (loss / 2).backward()
            se (passo + 1) divisivel por 2 ou este e o ultimo batch:
                atualizar_decoder(otimizador)
                zerar_gradientes(otimizador)

        decoder.eval()
        loss_validacao <- media da MSE para imagens normais de val_loader:
            alvo <- extrator(imagens)
            reconstruido <- decoder(alvo)  # eval desativa jitter
            loss_validacao <- MSE(reconstruido, alvo)
        scheduler.step()

        se loss_validacao < melhor_loss:
            salvar_checkpoint(decoder)
            melhor_loss <- loss_validacao
            epocas_sem_melhora <- 0
        senao:
            epocas_sem_melhora <- epocas_sem_melhora + 1
            se epocas_sem_melhora == 5:
                parar

    retornar melhor_checkpoint
```

Durante validacao, nao aplicar jitter; manter a mascara local ativa somente
para a versao completa. A implementacao esta em `src/reconstruction_decoder.py`;
`tests/test_reconstruction_decoder.py` inclui um teste de reducao da loss em
um unico batch, e `scripts/smoke_test_decoder.py` executa o decoder completo
em uma imagem normal de uma categoria. Na implementacao, a acumulacao de gradiente deve
normalizar a loss pelo numero de passos acumulados e tambem tratar o ultimo
batch incompleto.

## Pseudocodigo de inferencia e avaliacao

```text
funcao INFERIR(imagens, modelo):
    z <- modelo.extrator(imagens)               # layer1-3; somente imagem entra
    z_reconstruido <- modelo.decoder(z)
    mapa_grade <- norma_L2_canais(z - z_reconstruido)
    mapa_pixel <- interpolar_bilinear(mapa_grade, tamanho=imagem_original)
    # Suavizacao Gaussiana e opcional; manter valores brutos para metricas.
    escore_imagem <- aggregate_anomaly_maps(
        mapa_pixel, method="topk_mean", top_fraction=0.01
    )
    retornar escore_imagem, mapa_pixel

funcao AVALIAR(test_loader, modelo):
    para imagem, mascara, label, categoria em test_loader:
        escore, mapa <- INFERIR(imagem, modelo)
        guardar(escore, mapa, label, mascara, categoria)

    calcular AUROC de imagem usando escores e labels
    calcular AUROC de pixel e AUPRO usando mapas e mascaras
    reportar metricas por categoria e media macro
```

O teste oficial deve ser executado somente apos selecionar checkpoints e
congelar configuracoes. `mascara`, `label` e `categoria` permanecem fora de
`INFERIR`; sao metadados para avaliacao final. Redimensionar mascaras com
interpolacao nearest. Comparar baseline e versao completa sob o mesmo
protocolo.

Em `src/anomaly_scoring.py`, `topk_mean` (media dos 1% maiores pixels) e a
agregacao principal definida antes do teste: e menos sensivel a um pixel isolado
do que o maximo. A alternativa `max` aplica suavizacao Gaussiana 5 x 5
(`sigma=1`) antes de tomar o maior valor; ela fica disponivel para comparacao
sem escolher configuracao com base no teste.

O mapa espacial e calculado em `src/anomaly_maps.py` como norma L2 por posicao,
ampliado por interpolacao bilinear para a dimensao original/transformada da
imagem e retornado como `float32` com forma `(B,H,W)`. Suavizacao Gaussiana e
opcional. `normalize_map_for_visualization` destina-se somente a visualizacao;
calculo de metricas e agregacao usam os valores brutos. O script
`py -3.9 scripts/visualize_anomaly_maps.py` exporta exemplos lado a lado com as
mascaras. O decoder nessa previa esta sem treino, portanto a imagem demonstra
somente o fluxo e o alinhamento, nao a qualidade do detector.
