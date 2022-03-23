# Trabalho 1 - Aprendizado de Máquina 2
Alunos:
- Guilherme Arcencio
- Júlia Sato
- Vinicius Martins

Este repositório conta com a minha parte do trabalho 1 da disciplina de Aprendizado de Máquina 2, no qual foi elaborado um algoritmo no esquema ensemble para adquirir a melhor acurácia em três datasets diferentes.

O dataset apresentado nesta parte foi gerado para modelar resultados experimentais. Cada exemplo é classificado como tendo a balance scale tip à direita, à esquerda ou balanceada.

Os atributos são:

- Peso à esquerda (l_weight): 1, 2, 3, 4 e 5

- Distância à esquerda (l_dist): 1, 2, 3, 4 e 5

- Peso à direita (r_weight): 1, 2, 3, 4 e 5

= Distância à direita (r_dist): 1, 2, 3, 4 e 5

- E, por fim, a classe (bal_class), que pode ter os seguintes valores: L (esquerda), B (balanceada) e R (direita).

O dataset também apresenta um dado importante mas não exatamente em instanciamento: a balança só estará equilibrada quando os valores multiplicados de peso e distância de cada lado estejam iguais, ou seja, (l_weight * l_dist) = (r_weight * r_dist) no caso dos valores balanceados.

Dataset disponível no link: https://archive.ics.uci.edu/ml/datasets/Balance+Scale

Trabalho completo com os outros dois datasets feitos pelos outros dois membros do grupo: https://colab.research.google.com/drive/1Qaee8I8jKbrQEbS-KxoTwjM61YyeVn7C?usp=sharing
