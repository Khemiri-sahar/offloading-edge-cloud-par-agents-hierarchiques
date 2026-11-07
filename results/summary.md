Resultats sur 20 episodes par charge (moyenne, IC95 entre parentheses), charge : data/alibaba_batch_task.csv.
Device/Edge/Cloud = lieu d'execution ; Transfert = part des taches envoyees a l'edge voisin.

### Charge x0.5

| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |
|---|---|---|---|---|---|---|
| Aleatoire | 1.47 (0.46) | 90.8 (2.2) | 0.89 (0.17) | 34/33/34 | 0 | 0.0 |
| Toujours edge | 1.42 (0.71) | 84.3 (4.7) | 0.49 (0.09) | 0/100/0 | 0 | 0.0 |
| Greedy niveau le plus proche | 0.99 (0.18) | 99.1 (1.4) | 1.09 (0.15) | 76/22/3 | 0 | 7.8 |
| EDF-like (vue globale) | 0.47 (0.14) | 99.0 (1.3) | 0.30 (0.07) | 4/81/15 | 21 | 11.0 |
| RL centralise (flat) | 0.55 (0.18) | 98.5 (1.8) | 0.26 (0.10) | 9/58/33 | 3 | 11.0 |
| RL hierarchique | 0.54 (0.17) | 98.9 (1.4) | 0.24 (0.05) | 15/56/29 | 50 | 13.3 |

### Charge x1.0

| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |
|---|---|---|---|---|---|---|
| Aleatoire | 2.11 (0.76) | 86.5 (2.9) | 0.89 (0.17) | 34/33/34 | 0 | 0.0 |
| Toujours edge | 2.94 (1.89) | 67.2 (7.8) | 0.49 (0.09) | 0/100/0 | 0 | 0.0 |
| Greedy niveau le plus proche | 1.02 (0.22) | 98.9 (1.8) | 1.02 (0.13) | 69/24/7 | 0 | 4.9 |
| EDF-like (vue globale) | 0.52 (0.17) | 98.6 (1.9) | 0.31 (0.07) | 7/77/17 | 21 | 11.0 |
| RL centralise (flat) | 0.62 (0.23) | 98.1 (2.1) | 0.29 (0.11) | 12/57/32 | 6 | 11.0 |
| RL hierarchique | 0.62 (0.22) | 98.1 (2.0) | 0.26 (0.05) | 17/55/28 | 47 | 9.8 |

### Charge x2.0

| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |
|---|---|---|---|---|---|---|
| Aleatoire | 3.17 (1.17) | 77.9 (4.0) | 0.89 (0.17) | 34/33/33 | 0 | 0.0 |
| Toujours edge | 6.68 (3.80) | 40.4 (8.2) | 0.49 (0.09) | 0/100/0 | 0 | 0.0 |
| Greedy niveau le plus proche | 1.15 (0.37) | 98.0 (2.6) | 0.92 (0.11) | 58/25/16 | 0 | 4.0 |
| EDF-like (vue globale) | 0.59 (0.24) | 97.9 (2.7) | 0.35 (0.09) | 14/66/20 | 21 | 11.0 |
| RL centralise (flat) | 0.72 (0.30) | 96.9 (2.9) | 0.34 (0.13) | 18/52/31 | 10 | 11.0 |
| RL hierarchique | 0.71 (0.29) | 97.5 (2.1) | 0.29 (0.06) | 22/50/28 | 40 | 7.6 |

### Charge x3.0

| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |
|---|---|---|---|---|---|---|
| Aleatoire | 3.97 (1.47) | 71.6 (5.2) | 0.90 (0.17) | 34/33/33 | 0 | 0.0 |
| Toujours edge | 9.86 (4.40) | 25.4 (7.4) | 0.49 (0.09) | 0/100/0 | 0 | 0.0 |
| Greedy niveau le plus proche | 1.17 (0.33) | 97.6 (2.6) | 0.85 (0.09) | 51/23/25 | 0 | 4.1 |
| EDF-like (vue globale) | 0.73 (0.35) | 96.8 (3.1) | 0.37 (0.10) | 21/55/25 | 19 | 11.0 |
| RL centralise (flat) | 0.90 (0.42) | 95.5 (3.5) | 0.40 (0.15) | 24/46/30 | 11 | 11.0 |
| RL hierarchique | 0.89 (0.41) | 96.0 (2.3) | 0.31 (0.07) | 28/44/28 | 32 | 6.5 |

### Charge x4.0

| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |
|---|---|---|---|---|---|---|
| Aleatoire | 4.61 (1.67) | 65.7 (5.2) | 0.90 (0.17) | 34/34/33 | 0 | 0.0 |
| Toujours edge | 12.35 (4.71) | 18.2 (6.3) | 0.49 (0.09) | 0/100/0 | 0 | 0.0 |
| Greedy niveau le plus proche | 1.31 (0.52) | 96.6 (3.3) | 0.80 (0.09) | 45/21/34 | 0 | 4.4 |
| EDF-like (vue globale) | 0.88 (0.46) | 94.8 (4.9) | 0.41 (0.12) | 26/45/29 | 16 | 11.0 |
| RL centralise (flat) | 1.03 (0.49) | 94.2 (3.2) | 0.46 (0.18) | 28/40/32 | 11 | 11.0 |
| RL hierarchique | 1.09 (0.53) | 94.5 (2.7) | 0.34 (0.08) | 33/40/27 | 26 | 5.8 |

