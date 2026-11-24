# Decisao de implementacao: UniAD

**Data da pesquisa:** 2026-11-23
**Decisao:** **B — implementar uma versao reduzida e propria, inspirada no artigo UniAD.**

## Contexto e criterio

O objetivo do projeto e treinar um detector unico com imagens normais das 15
categorias MVTec AD, sem fornecer a categoria ao modelo, e produzir mapas de
anomalia por pixel. A implementacao deve integrar os DataLoaders do repositorio,
a entrada candidata de 512 x 512 e o ambiente de notebook/Colab, com poucas
dependencias e atribuicao clara de artigos, bibliotecas e codigo consultado.

## Candidatos avaliados

| Candidato | Licenca e requisitos | Treino conjunto e mapa por pixel | Prós | Contras / compatibilidade |
|---|---|---|---|---|
| [Implementacao oficial UniAD](https://github.com/zhiyuanyou/UniAD) | Apache-2.0. `requirements.txt` fixa PyTorch 1.5.0, torchvision 0.6.0+cu90, NumPy 1.19.5, OpenCV 3.4.10.37 e outras dependencias antigas. | Sim. O README reporta um modelo unificado para todas as categorias MVTec AD. A configuracao reporta uma saida `pred` espacial, calculada pela discrepancia de features e ampliada a partir da grade de features. O metadado de categoria e usado para organizar resultados, nao como sinal para calcular essa discrepancia. | Implementacao fiel ao artigo; disponibiliza treinamento, avaliacao e resultados de referencia publicados; CUDA/GPU prevista. | A instalacao fixa versoes antigas e acopladas a CUDA 9; config MVTec usa 224 x 224, batch 8 e ate 1000 epocas. O codigo usa `.cuda()` diretamente na injecao de jitter e na mascara de atencao. Isso requer adaptacao de dependencias, dispositivos, treino e entrada para este projeto. |
| [anomalib](https://github.com/open-edge-platform/anomalib), incluindo [catalogo de modelos](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html) | Apache-2.0. A versao publicada consultada (2.7.1) requer Python >= 3.10; os extras oficiais de instalacao PyTorch (`cpu`/`cu126`) na linha principal declaram PyTorch >= 2.10. | **Nao inclui UniAD** no catalogo consultado. Inclui UniNet, um modelo distinto, que produz mapas de anomalia. Embora se descreva como unificado/multidominio, o exemplo documentado do conjunto MVTec recebe `data.category` e seu datamodule `MVTecAD` usa `category="bottle"` como padrao; a integracao documentada e por categoria, nao e um substituto direto para o treino combinado requerido. | Biblioteca ativa com componentes, treinamento, predicao e metricas prontos; Apache-2.0; UniNet oferece mapa espacial e pode ser avaliado como outro algoritmo. | Nao contem a implementacao UniAD. Usar UniNet mudaria o metodo do projeto; combinar as 15 categorias exigiria adaptar o datamodule e validar o protocolo. Nao pode ser instalado no Python 3.9/PyTorch 2.8 deste ambiente sem mudar o ambiente. |

Fontes primarias adicionais:

- [Artigo UniAD, NeurIPS 2022](https://arxiv.org/abs/2206.03687)
- [Configuração MVTec da implementacao oficial](https://github.com/zhiyuanyou/UniAD/blob/main/experiments/MVTec-AD/config.yaml)
- [Codigo oficial do decodificador e mapa espacial](https://github.com/zhiyuanyou/UniAD/blob/main/models/reconstructions/uniad.py)
- [Dependencias oficiais fixadas](https://github.com/zhiyuanyou/UniAD/blob/main/requirements.txt)
- [Licenca Apache-2.0 do repositorio oficial](https://github.com/zhiyuanyou/UniAD/blob/main/LICENSE)
- [Documentacao da integracao MVTecAD do anomalib](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
- [Documentacao do UniNet no anomalib](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/uninet.html)
- [PyPI: anomalib 2.7.1](https://pypi.org/project/anomalib/2.7.1/)
- [Google Colab: recursos e runtimes](https://research.google.com/colaboratory/faq.html)

O README do repositorio oficial tambem reconhece uso de codigo de DETR e
EfficientNet-PyTorch. Como nenhum desses componentes sera copiado para este
projeto, eles nao sao dependencias da implementacao decidida; caso isso mude,
registrar e citar os respectivos repositorios e licencas.

## Instalacao e exemplo minimo

O teste local de disponibilidade de binarios foi executado com Python 3.9 e
`pip download --no-deps --only-binary=:all: torch==1.5.0`: **falhou**, pois nao
existe distribuicao binaria compativel oferecida pelo indice para esse
ambiente. A lista apresentada por pip comeca em PyTorch 1.7.1. Isso confirma
incompatibilidade local com o pin principal do UniAD; nao e uma execucao do
projeto oficial nem um teste de Colab.

Os metadados publicados de anomalib 2.7.1 requerem Python >= 3.10. O ambiente
local consultado e Python 3.9 com PyTorch 2.8.0+cpu; portanto, nao foi feita
instalacao ou execucao do exemplo do anomalib. Alem disso, a biblioteca nao
registra UniAD como modelo. UniNet nao foi executado: sua instalacao nao e
compativel com o ambiente local e nao corresponde ao metodo-alvo sem adaptar o
carregamento MVTec.

**Colab: nao validado.** Esta sessao nao tem um runtime Google Colab conectado.
Nao se deve interpretar os resultados de compatibilidade local acima como
teste de instalacao em Colab. Antes do treinamento, executar no notebook uma
celula de instalacao, registrar Python/PyTorch/CUDA do runtime e completar um
forward e backward com um batch pequeno. A versao do runtime de Colab e mutavel;
nao presumir que corresponda ao Python ou CUDA local.

Como nenhum candidato e executavel diretamente e aderente ao protocolo neste
ambiente, nao foi selecionado um modelo terceiro. O notebook
`notebooks/category_pipeline_checkpoint.ipynb` executa o fluxo DataLoader,
extrator, decoder, mapa e escore em uma amostra defeituosa de `bottle`, usando
o decoder sem treino. O teste comprova integracao e dimensoes, nao qualidade do
detector nem compatibilidade com Colab.

## Decisao: B

Implementar no repositorio uma versao reduzida original, em vez de portar a
implementacao oficial. A especificacao exata do modelo minimo, da versao
completa, do baseline e do protocolo comparativo esta em
[`model-baseline-specification.md`](model-baseline-specification.md). A
especificacao escolhe Wide-ResNet-50-2 congelado, features alinhadas de layer1
a layer3, entrada 512 x 512 e
um decodificador com consultas aprendidas; mask e jitter ficam fora do baseline
e entram somente na versao completa. O modelo recebe tensores de imagem,
nunca nomes de categoria, labels ou mascaras.

Esta decisao **nao** afirma que o resultado reproduz exatamente UniAD ou seus
resultados. Nenhum arquivo de codigo do repositorio UniAD ou do anomalib foi
copiado. Os artigos, as implementacoes consultadas e bibliotecas efetivamente
usadas devem permanecer citados no notebook e no relatorio. Seguir Apache-2.0
caso codigo de qualquer um desses repositorios seja incorporado futuramente;
registrar tambem arquivo, commit/release e alteracoes.

## Cronograma revisado (estimativa)

| Etapa | Esforco | Entrega / criterio de conclusao |
|---|---:|---|
| Preparar e registrar runtime Colab | 0,5 dia — pendente | Registrar Python, PyTorch, CUDA e VRAM; instalar dependencias; executar um forward e backward com batch pequeno. |
| Implementar prototipo e checkpoint em uma categoria | Concluido localmente | Testes de forward/backward e reducao da loss; notebook executa DataLoader -> extrator -> decoder -> mapa -> escore e salva figura para `bottle`. Decoder ainda sem treino; checkpoint valida integracao, nao qualidade. |
| Treino curto do baseline em 1 categoria | Concluido localmente | Duas epocas de `bottle` em CPU a 256 x 256, batch 1; MSE caiu em treino e validacao, e o patch sintetico teve maior discrepancia na regiao alterada. Teste oficial nao usado. Resultados e limites em `docs/experiments/quick-validation-bottle.md`; isso nao substitui o preflight GPU. |
| Treinar e comparar baseline/completo | 1-4 dias — apos preflight | Treinar nas imagens normais combinadas; avaliar teste oficial uma vez, com metricas de imagem e pixel por categoria. Ajustar tempo/epocas conforme VRAM e duracao observadas. |
| Comparar resolucao, analisar erros e documentar | 0,5 dia — pendente | Comparar resolucoes; inspecionar mapas do decoder treinado nos menores defeitos; atualizar notebook, citacoes e relatorio. |

**Estimativa restante a partir deste checkpoint:** 2-5 dias de trabalho, dos
quais 1-4 dias dependem do treino/avaliacao em GPU e cerca de 0,5 dia de runtime
e validacao de resolucao. O checkpoint local passou em CPU, mas VRAM e
throughput de GPU continuam sem medicao. O teste oficial deve permanecer
intocado ate os checkpoints e configuracoes serem escolhidos pela validacao.
As epocas sao escolhas experimentais, nao equivalem aos 1000 epochs do codigo
oficial.

## Atribuicao de terceiros

| Item | Uso neste projeto | Citacao |
|---|---|---|
| PyTorch | Framework usado pelo DataLoader e previsto para o modelo proprio. | Paszke et al., 2019; versao efetiva registrada no notebook. |
| NumPy e Pillow | Leitura/pre-processamento do DataLoader. | Harris et al., 2020; documentacao Pillow; versoes efetivas no notebook. |
| Artigo UniAD e repositorio oficial | Inspiracao metodologica e implementacao consultada; codigo nao copiado. | You et al., 2022; repositorio oficial UniAD, Apache-2.0. |
| anomalib / UniNet | Candidato avaliado, nao adotado e sem codigo reutilizado. | Open Edge Platform, anomalib 2.7.1 e documentacao UniNet. |
