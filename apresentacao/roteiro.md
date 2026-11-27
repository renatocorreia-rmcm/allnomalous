# Roteiro sugerido para a dupla

Pedro apresenta os slides 1–4. Mostre como uma peça sem categoria conhecida chega à inspeção. Use o histograma para explicar por que 499 defeitos pequenos motivam avaliar localização. No diagrama, siga uma imagem do extrator até o mapa e o escore. Termine explicando como os checkpoints foram escolhidos.

**Transição para Renato.** “Com os checkpoints fixados, o que aconteceu quando os dois modelos viram as mesmas peças?”

Renato apresenta os slides 5–8. Compare primeiro as médias macro. No zíper, aponte a borda detectada e os dentes partidos que passaram despercebidos. Em seguida, use as distribuições das peças normais para explicar como um limiar global pode favorecer uma categoria. Encerre com o experimento pareado que permitiria avaliar a diferença de 0,19 ponto de AUPRO.

## Perguntas para o ensaio

- **Por que dispensar a categoria?** Um único detector pode inspecionar peças de tipos diferentes mesmo quando a linha não informa o tipo. O deslocamento das distribuições normais dificulta definir um alarme comum.
- **Como o teste participou da seleção?** Os checkpoints foram escolhidos com imagens normais de validação. Houve inspeção visual prévia de exemplos do teste. O P95 apresentado foi calculado após a avaliação e serve para interpretar os erros.
- **Como interpretar AUROC alto por pixel e baixo por imagem?** A ordenação de pixels mistura regiões com e sem defeito. O escore agregado da peça também responde a contornos normais intensos.
- **O modelo completo é superior?** Os resultados macro foram próximos. Uma semente e orçamentos diferentes impedem atribuir causalmente a diferença aos componentes adicionados.

Após confirmar a duração com o professor, cronometrem uma passagem completa e uma segunda com os ajustes. Registrem o tempo real de cada parte no cartão do ensaio.
