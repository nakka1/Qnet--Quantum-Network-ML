# RTL do compensador

Gerado de `ridge (Wiener) · 16 bits, CORDIC 8` por `qnet_autopilot.deploy.verilog.emit_verilog`.

## Rodar o testbench

```bash
iverilog -o qnet_tb qnet_tb.v qnet_mac.v && vvp qnet_tb
```

Ele imprime `PASS` ou o índice exato da primeira palavra que divergiu.
Os vetores são os valores intermediários do próprio modelo de referência em `qnet_autopilot/deploy/fixed_point.py`, então uma divergência é bug do RTL e não desacordo de modelagem.

**Este testbench nunca foi executado.** O contêiner onde o pacote foi desenvolvido não alcança nem o índice de pacotes nem o repositório da distribuição, então não há simulador HDL nele. A fidelidade do RTL ao modelo de referência é garantida por construção — mesmas quatro operações, na mesma ordem — e não por co-simulação. Rode o bench antes de confiar nele.

## Parâmetros por camada

| camada | entradas | saídas | formato entrada | formato peso | formato saída | shift | acumulador | ReLU |
|---|---|---|---|---|---|---|---|---|
| 0 | 48 | 3 | Q3.12 | Q8.7 | Q11.4 | 15 | 38 bits | não |

## Dimensionamento

- 144 multiplica-acumula por inferência; com um MAC serial a 144 ciclos, são 1 us a 100 MHz, contra um período de controle de 2 ms.
- ROM de parâmetros: 0.3 KiB a 16 bits.
- 15 logaritmos de SO(3) por passo, 16 passagens de CORDIC no total — este é o bloco caro, não os MACs.

O gargalo de tempo medido em software é o mesmo: os logaritmos levam 72% do caminho de dados e a rede 15%. Um port que acelere só a rede acelera a parte barata.
