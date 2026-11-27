# Avaliação final no MVTec AD

| Categoria | Baseline AUROC imagem | Baseline AUROC pixel | Baseline AUPRO | Completo AUROC imagem | Completo AUROC pixel | Completo AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 96,35 | 91,09 | 84,10 | 96,51 | 91,02 | 84,10 |
| cable | 71,68 | 73,33 | 65,75 | 71,01 | 73,12 | 65,88 |
| capsule | 64,62 | 93,04 | 84,32 | 65,10 | 93,12 | 84,56 |
| carpet | 89,73 | 94,88 | 86,28 | 89,17 | 94,50 | 85,81 |
| grid | 85,05 | 94,73 | 81,37 | 85,71 | 95,20 | 82,89 |
| hazelnut | 97,86 | 95,61 | 90,28 | 98,00 | 95,67 | 90,44 |
| leather | 94,53 | 97,51 | 93,95 | 94,16 | 97,51 | 94,02 |
| metal_nut | 57,33 | 81,50 | 63,52 | 56,94 | 81,34 | 63,52 |
| pill | 66,18 | 86,07 | 86,22 | 66,83 | 86,19 | 86,35 |
| screw | 51,22 | 95,46 | 84,24 | 51,71 | 95,54 | 84,48 |
| tile | 94,08 | 87,19 | 68,45 | 93,80 | 86,82 | 67,95 |
| toothbrush | 80,00 | 92,35 | 72,28 | 80,56 | 92,59 | 72,87 |
| transistor | 56,29 | 65,99 | 43,38 | 56,17 | 66,06 | 43,77 |
| wood | 97,63 | 93,05 | 91,96 | 97,54 | 93,00 | 91,93 |
| zipper | 73,45 | 94,38 | 82,25 | 73,69 | 94,50 | 82,67 |
| macro | 78,40 | 89,08 | 78,56 | 78,46 | 89,08 | 78,75 |

Resultados em porcentagem. A média macro atribui o mesmo peso a cada uma das 15 categorias. AUROC de imagem por postos, AUROC de pixel e AUPRO por histogramas monotônicos de 2.048 bins. A AUPRO é normalizada até FPR de 0,30, com regiões de conectividade 4.

## Protocolo e rastreabilidade

Commit do avaliador: 34f28c571d55291a12fc06d63aacdb4dac503abd.
Entrada RGB 512 × 512, normalização ImageNet, WideResNet50-2 congelada, discrepância L2, interpolação bilinear e média do 1% superior para o escore de imagem.

- **baseline**, época 109, SHA-256 `a75f06c0344f4863f4c9114f714eeab44cee42c56fcb1d099a1737c60a6ebdcb`, commit de treino `a9b66186d8b66a05a3fa3a2caf2524a75a0231ae`.
- **complete**, época 315, SHA-256 `fe06f45a2caba1d0d9058048e6c10fa6f77c259a492ac2a198d3375f12bee395`, commit de treino `1e80dfedb2219ead5e0720b0cb87857da6156a08`.

## Seleção dos casos e legendas

Limiar diagnóstico de imagem: P95 dos escores das normais do modelo completo (32.9546). Limiar de mapa: P99 dos pixels das normais, amostrados a cada 8 pixels em cada dimensão (28.6670). Esses limiares descrevem os erros nesta rodada e devem ser calibrados em um conjunto independente para uso operacional.

Em cada grupo, o VP é o defeito detectado com maior IoU, o FN é o defeito não detectado com menor escore e o FP é a imagem normal com maior escore. Em empates, vale a ordem do caminho do arquivo. Cada figura usa uma escala L2 comum e a sobreposição tem o contorno ciano da anotação.

No grupo texturas, o limiar global não produziu FP. Para ilustrar um alarme relativo ao grupo, a figura usa a normal de maior escore acima do P95 das normais desse grupo (24.0983). As contagens abaixo continuam usando o limiar global.

- **objetos · VP** — `zipper/test/fabric_border/001.png`. Escore 34.134, IoU 0.587, máscara em 0.970% da imagem.
- **objetos · FN** — `zipper/test/split_teeth/009.png`. Escore 21.999, IoU 0.000, máscara em 2.123% da imagem.
- **objetos · FP** — `screw/test/good/002.png`. Escore 37.080, IoU 0.000, máscara em 0.000% da imagem.
- **texturas · VP** — `wood/test/hole/003.png`. Escore 33.774, IoU 0.588, máscara em 1.240% da imagem.
- **texturas · FN** — `leather/test/color/004.png`. Escore 20.897, IoU 0.000, máscara em 0.688% da imagem.
- **texturas · FP (P95 do grupo)** — `grid/test/good/012.png`. Escore 25.964, IoU 0.000, máscara em 0.000% da imagem.

## Erros e limitações

- Menores valores de AUROC de imagem no modelo completo: screw (51.71%), transistor (56.17%), metal_nut (56.94%).
- Menores valores de AUROC de pixel no modelo completo: transistor (66.06%), cable (73.12%), metal_nut (81.34%).
- Menores valores de AUPRO no modelo completo: transistor (43.77%), metal_nut (63.52%), cable (65.88%).
- **Objetos** — 781/876 FN e 24/334 FP. Defeitos com máscara <1%: 286/343 FN; máscaras ≥1%: 495/533 FN. Em 0/24 FP, ao menos metade dos pixels acima do limiar fica nos 32 pixels próximos à borda da imagem.
- **Texturas** — 369/382 FN e 0/133 FP. Defeitos com máscara <1%: 153/156 FN; máscaras ≥1%: 216/226 FN. Em 0/0 FP, ao menos metade dos pixels acima do limiar fica nos 32 pixels próximos à borda da imagem.
- Os rótulos disponíveis descrevem defeito e máscara. Variações de pose, bordas do objeto e reflexos são hipóteses a verificar visualmente nos mapas; a contagem de borda acima se refere à borda da imagem.
- A grade de features é 32 × 32 e os mapas são interpolados a 512 × 512. Isso limita a localização fina. O extrator foi pré-treinado em ImageNet. Ambos os treinos usam uma única semente. Os checkpoints diferem em duração e critério de seleção, de modo que as métricas comparam as versões treinadas e não isolam o efeito da máscara e do jitter.
- A validação de treino tinha somente imagens normais. Houve inspeção visual prévia de amostras do teste e ela deve ser registrada ao interpretar a avaliação oficial.
- Compatibilidade corrigida antes da rodada: integração trapezoidal via SciPy para funcionar também com NumPy recente. Os mapas foram arquivados em precisão float32 para preservar a mesma precisão usada nas métricas.

### Leitura visual dos erros

A mancha em leather ocupa 0,69% da imagem e teve resposta fraca sobre a textura do couro. O defeito nos dentes do zíper também ficou sem sobreposição ao limiar de pixel, embora ocupe 2,12%. O tamanho da máscara sozinho é insuficiente para explicar ambos os casos.

O falso alarme do grupo de texturas aparece na periferia de uma imagem normal de grid. No parafuso normal, o mapa destaca a cabeça, a rosca e a ponta. Essas ativações coincidem com contornos do objeto e variação de brilho. O teste não isolou pose, iluminação ou reflexos, e os mapas não permitem atribuir a falha a um desses fatores.

## Discussão

Em comparação com o baseline, o modelo completo variou +0.06 p.p. em AUROC de imagem, -0.0001 p.p. em AUROC de pixel e +0.19 p.p. em AUPRO macro. As medianas dos escores normais foram de 21.11 (leather) a 32.60 (toothbrush), o que quantifica o deslocamento de escala entre categorias e ajuda a interpretar o efeito de aplicar um limiar global. Os mapas e a análise por tamanho de máscara mostram por que a ordenação de imagens e a localização espacial devem ser avaliadas separadamente.

## Registro operacional

O job 33437 executou uma passagem por modelo sobre as 1.725 imagens. As submissões preliminares terminaram antes da leitura do teste, por falta ou instalação incompleta do SciPy e por uma asserção antiga de float16 que foi atualizada para float32. As dez verificações passaram no commit congelado. As figuras e o diagnóstico foram gerados dos arquivos salvos, sem novas inferências. A figura de texturas registra explicitamente o limiar ilustrativo por grupo.

Mapas float32, máscaras compactadas, escores, rótulos e caminhos dos dois modelos ficaram em `~/allnomalous-runs/final_eval_315/results/` no Apuana, nos arquivos `baseline_predictions.npz` e `complete_predictions.npz`. O manifesto `final_test_run.json` registra os checkpoints e os hashes usados.
