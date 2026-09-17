# Artigo

`artigo.tex` — o artigo sobre o `qnet_autopilot`, no formato de artigo da SBC,
em português, com resumo em inglês.

O conteúdo é exclusivamente sobre este pacote. O PDF do TCC que serviu de
referência foi usado **apenas como modelo de formatação** — layout, estrutura de
título e resumo, estilo de citação `[Autor ano]` e formato da lista de
referências. Nada do assunto daquele trabalho entrou aqui; as referências em
comum (Bennett, Briegel, Preskill, Krutyanskiy) são as canônicas da área e foram
reconferidas contra a fonte primária de forma independente.

Estrutura: Introdução, Trabalhos Relacionados, O Substrato Físico, Arquitetura
do Sistema, As Quatro Capacidades, Metodologia de Avaliação, Resultados, Três
Parâmetros que a Avaliação Corrigiu, Limitações, Conclusão.

```bash
python paper/make_figures.py          # gera figs/*.pdf a partir de runs/
cd paper && pdflatex artigo && pdflatex artigo
```

Duas rodadas de `pdflatex` são necessárias porque as referências cruzadas de
figura e tabela só resolvem na segunda.

## Sobre o `sbc-template.sty` deste diretório

O pacote oficial da SBC não está disponível no contêiner onde isto foi
desenvolvido (sem acesso a CTAN), e um artigo que não compila não é um
entregável. O `sbc-template.sty` aqui reproduz o layout do modelo — A4, Times
12pt, título e autores centralizados, blocos de *Abstract* e *Resumo* recuados
em itálico, seções numeradas com ponto, legendas em corpo menor com rótulo em
negrito, citações `[Autor ano]` e bibliografia sem rótulos com recuo pendente.

Os comandos usados (`\title`, `\author`, `\address`, `\email`, `abstract`,
`resumo`, `\keywords`) têm a mesma assinatura do pacote oficial, então **para
submeter, basta apagar este arquivo e pôr o oficial no lugar**. Duas ressalvas
que o oficial resolve sozinho: este ambiente não traz o arquivo de idioma do
babel para português, então `\figurename` e `\tablename` são fixados à mão; e
`\cite` está redefinido como `\citep` do natbib para produzir `[Autor ano]`.

## Referências

Todas as 33 referências foram conferidas contra a fonte primária. Três
referências que apareciam em versões anteriores deste trabalho **não existem** e
foram substituídas por trabalhos reais sobre o mesmo assunto:

| referência inexistente | substituída por |
|---|---|
| Dietsche, Dür & Dunjko, *npj QI* 8:14 (2022) | Wallnöfer et al., *PRX Quantum* 1:010301 (2020); Roik et al., *QIP* 23:89 (2024) |
| Cao, Yehia & Tura, *IEEE TQE* 4 (2023) | Le & Nguyen, *ICC* 2022, p. 395–400 |
| Zhuang & Zhang, *npj QI* 10:14 (2024) | Krastanov et al., *Quantum* 3:123 (2019); Chittoor & Simeone, *Entropy* 25(2):352 (2023) |

Uma quarta foi corrigida em atribuição: a polarimetria a 4 kHz em cabo
para-raios é de **Peterson Jr., Leo & Rochford** (OFC 2004), não de "Peterson,
Cyr & Nyman". A correção foi propagada para
`qnet_autopilot/physics/field_calibration.py`.
