# Avaliação final

O job 33437 avaliou os dois checkpoints sobre as mesmas 1.725 imagens do teste oficial. O [relatório](relatorio_avaliacao.md) descreve o protocolo, as inspeções anteriores, os limiares diagnósticos e as limitações.

## Checkpoints e históricos

| Job | Variante | Checkpoint | Histórico |
|---|---|---|---|
| 32896 | Baseline | [Época 109](../../cluster_results/job_32896/best_decoder.pt) | [Até a época 111](../../cluster_results/job_32896/loss_history.csv) |
| 33358 | Completo com early stopping | [Época 315](../../cluster_results/job_33358/selected_decoder.pt) | [Até a época 385](../../cluster_results/job_33358/loss_history.csv) |
| 33292 | Completo exploratório | [Época 520, retomável](../../cluster_results/job_33292/last_checkpoint.pt) | [Até a época 520](../../cluster_results/job_33292/loss_history.csv) |

O job 33292 não participou da avaliação oficial. A configuração de treino está junto a cada checkpoint. O [manifesto da rodada](final_test_run.json) registra os hashes oficiais.

## Resultados e reanálise

- [Métricas oficiais](final_metrics.csv), [escores dos dois modelos por imagem](escores_ambos.csv) e [diagnóstico do modelo completo](diagnostico_por_imagem.csv)
- [Curvas de loss dos treinos da variante completa](loss_modelo_completo.png)
- [Casos selecionados](casos_selecionados.csv), [relatório](relatorio_avaliacao.md) e [tabela LaTeX](tabela_final.tex)
- [Manifesto dos arquivos brutos](npz_manifest.json), com nomes, caminhos, tamanhos e SHA-256

Os dois NPZ brutos permanecem no Apuana e contêm mapas float32, máscaras compactadas, escores, caminhos e rótulos. Para regenerar os gráficos sem nova inferência, copie os NPZ, o CSV oficial e o manifesto da rodada para um diretório de trabalho. Execute os comandos abaixo a partir da raiz do repositório.

```bash
python -m pip install -r requirements.txt
python scripts/report_final.py /caminho/para/artefatos /caminho/para/MVTec-AD
```

O relatório versionado inclui uma leitura visual complementar dos casos. As imagens do MVTec AD e as figuras derivadas seguem a [licença do dataset](../../license.txt).
