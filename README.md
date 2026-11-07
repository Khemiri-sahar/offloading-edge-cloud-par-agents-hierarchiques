# Projet 5 : offloading Edge/Cloud par agents hierarchiques

Mini-projet Futurs Reseaux Mobiles (INSAT RT5, 2026/2027).

Tout est en Python, sans simulateur Java : un simulateur MEC maison (Gymnasium)
a trois niveaux (device, edge, cloud/HPC), un mecanisme de decision par niveau
(DQN au device, DQN a l'edge, ordonnanceur EDF pondere au cloud), des baselines
et une evaluation reproductible sur une trace reelle Alibaba.

## 1. Probleme

En MEC (Multi-access Edge Computing), une tache de calcul nee sur un device
peut s'executer localement, sur un serveur edge proche, ou sur un cloud/HPC
distant (tres puissant mais plus lent a atteindre). Une decision « plate »,
prise par un agent centralise qui voit tout, oblige a remonter l'etat complet du
systeme a chaque tache et passe mal a l'echelle.

Ce projet propose une **hierarchie de decision** : chaque niveau decide avec
l'information dont il dispose localement, et les niveaux se coordonnent par des
signaux legers (beacons de charge, contre-pression) et par une recompense
commune (latence, energie, deadline).

## 2. Correspondance avec le sujet

| Demande du sujet | Implementation |
|---|---|
| Architecture a 3 niveaux device / edge / cloud-HPC | `core.py`, classe `Sim` |
| Device : observe la tache (taille, deadline, intensite), la batterie, la connectivite ; local ou escalade vers l'edge | `Sim.dev_obs`, `DeviceEnv`, agent DQN |
| Edge : execution locale ou offloading vers le cloud, equilibrage de charge | `Sim.edge_obs`, `EdgeEnv`, agent DQN a 3 actions (edge, cloud, edge voisin) |
| Cloud : coordinateur global, allocation HPC entre taches de plusieurs edges, EDF pondere, contre-pression | `Sim._cloud_schedule`, `_commit`, `cloud_backlog`, `plan(CLOUD)` |
| Recompense par niveau : latence + energie locale + penalite de deadline | `Sim.reward` |
| DQN/PPO pour device et edge, mecanisme classique au cloud | `train.py` (DQN stable-baselines3), EDF non appris au cloud |
| Trace publique (Alibaba / Google) re-echantillonnee | `data/alibaba_batch_task.csv`, `trace_loader.py`, `gen_tasks` |
| Latence reseau explicite device-edge et edge-cloud | `Config.lat_*`, `Config.bw_*`, `Sim.plan` |
| Comparaison a aleatoire, greedy niveau le plus proche, EDF, approche centralisee | `policies.py`, `evaluate.py` |
| Compromis coordination / surcout de communication | `control_values`, etude de la periode des beacons |
| Extension federee | non faite (optionnelle) |

## 3. Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Aucun GPU necessaire. Teste avec Python 3.12, Gymnasium 1.4,
stable-baselines3 2.9 et PyTorch 2.x.

## 4. Lancer (reproduire tous les resultats)

```bash
.venv/bin/python train.py      # entrainement complet, environ 5 minutes sur CPU
.venv/bin/python evaluate.py   # evaluation et figures dans results/, environ 1 minute
```

- Par defaut, entrainement et evaluation rejouent la trace Alibaba fournie
  (`data/alibaba_batch_task.csv`).
- Charge synthetique de Poisson : `train.py --trace ""` et `evaluate.py --trace ""`.
- Test rapide : `train.py --quick` puis `evaluate.py --episodes 3`.
- Autre graine d'entrainement : `train.py --seed 1`.
- Options de `train.py` : `--rounds`, `--edge-steps`, `--dev-steps`,
  `--flat-steps`, `--seed`, `--trace`, `--quick`.
- Options de `evaluate.py` : `--episodes`, `--trace`.

Les graines sont fixees : memes commandes, memes resultats.

## 5. Contenu du depot

| Fichier | Role |
|---|---|
| `core.py` | Simulateur : configuration, generation des taches, files d'attente, ordonnanceur EDF du cloud, energie, observations, recompenses, metriques |
| `envs.py` | Environnements Gymnasium : `DeviceEnv`, `EdgeEnv`, `FlatEnv` |
| `policies.py` | Baselines (aleatoire, toujours edge, greedy niveau le plus proche, EDF-like) et enveloppes des agents RL |
| `train.py` | Entrainement sequentiel des agents (DQN, stable-baselines3) |
| `evaluate.py` | Comparaison sur flux de taches identiques, tableaux et figures |
| `trace_loader.py` | Chargeur de trace (CSV `start_time,duration`) et extraction depuis Alibaba |
| `data/alibaba_batch_task.csv` | Sous-ensemble de la trace Alibaba cluster-trace-v2018 (`batch_task`) : 20 000 taches terminees, environ 20 h |
| `models/` | Modeles entraines : `device.zip`, `edge.zip`, `flat.zip` |
| `logs/` | Courbes d'apprentissage (`*.monitor.csv`) |
| `results/` | `results.csv`, `summary.md`, `beacon_study.csv`, figures PNG |
| `train_log.txt` | Journal du dernier entrainement |

## 6. Modele du systeme

### 6.1 Topologie et parametres (`Config` dans `core.py`)

| Parametre | Valeur |
|---|---|
| Devices | 6, CPU tire entre 0.7 et 1.5 Gcycles/s, batterie 150 J (niveau initial 40 a 100 %) |
| Edges | 2 serveurs de 6 Gcycles/s, file FIFO ; le device `d` est rattache a l'edge `d mod 2` |
| Cloud/HPC | 2 coeurs de 8 Gcycles/s, partages par tous les edges, EDF pondere |
| Lien device-edge | 5 ms + 30 Mbit/s multiplie par un fading tire par tache (0.3 a 1.2) |
| Lien edge-cloud | 150 ms + 50 Mbit/s |
| Lien edge-edge | 10 ms + 100 Mbit/s |
| Energie | device : 1.2 J/Gcycle, emission 0.8 W ; edge : 0.25 J/Gcycle, emission 3 W |
| Contre-pression cloud | refus si l'attente estimee depasse 1 s |
| Periode des beacons | 1 s |
| Episode | 300 taches ; taux de base 0.5 tache/s/device, multiplie par la charge (0.5 a 4) |

### 6.2 Charge de travail

**Trace Alibaba (par defaut).** `data/alibaba_batch_task.csv` contient les
20 000 premieres taches terminees du fichier `batch_task` de la trace Alibaba
cluster-trace-v2018 (colonnes gardees : instant de debut, duree). Pour chaque
episode, une **fenetre contigue** de 300 taches est rejouee : les ecarts entre
arrivees (rafales comprises) sont remis a l'echelle du taux d'arrivee vise, et
les durees sont converties en cycles (moyenne 2.4 Gcycles, bornes 0.1 a 20).
La taille de la tache et sa deadline sont derivees des cycles.

Pour regenerer le sous-ensemble depuis le fichier Alibaba brut :

```bash
.venv/bin/python trace_loader.py batch_task.csv data/alibaba_batch_task.csv
```

**Charge synthetique (option).** Arrivees de Poisson et trois types de taches :

| Type | Probabilite | Taille (Mbit) | Cycles (Gcycles) | Deadline (s) |
|---|---|---|---|---|
| Legere | 0.4 | 0.2 a 1 | 0.2 a 0.6 | 1 a 2 |
| Moyenne | 0.4 | 1 a 3 | 1 a 3 | 2 a 4 |
| Lourde | 0.2 | 3 a 8 | 4 a 10 | 4 a 8 |

### 6.3 Parcours d'une tache

```
device ──(local)──> execution sur le device (file FIFO)
   │
   └──(escalade)──> edge ──(local)────────> execution sur l'edge (FIFO)
                     │
                     ├──(edge voisin)─────> execution sur l'autre edge (FIFO)
                     │
                     └──(cloud)──> cloud/HPC : file EDF ponderee
                                     │ attente estimee > 1 s : refus,
                                     └─> retour a l'edge (aller-retour perdu)
```

Latence de bout en bout = transmission + propagation + attente + calcul (le
retour du resultat est neglige).

### 6.4 Les trois niveaux de decision

**Device (agent DQN, 2 actions : local / edge).** Observation (8 valeurs,
normalisees) : taille, cycles, deadline, batterie, attente de sa file locale,
debit de son lien, derniere charge diffusee par son edge, vitesse de son CPU.

**Edge (agent DQN, 3 actions : executer / cloud / edge voisin).** Observation
(6 valeurs) : taille, cycles, temps restant avant la deadline, attente de sa
file, derniere charge diffusee par l'edge voisin, derniere contre-pression du
cloud. Le transfert a l'edge voisin equilibre la charge entre les groupes de
devices.

**Cloud/HPC (coordinateur non appris).** Les taches de tous les edges partagent
les coeurs, servis en Earliest-Deadline-First pondere non preemptif : priorite
= arrivee + deadline relative / poids de l'edge d'origine (`edf_weights`, 1 par
defaut, soit EDF simple). Une tache dont l'attente estimee depasse 1 s est
refusee (contre-pression). Le cloud diffuse aussi son backlog aux edges.

**Signaux de coordination.** Une fois par seconde : charge de chaque edge vers
ses devices et vers l'edge voisin, backlog du cloud vers les edges.
L'information est donc perimee entre deux beacons.

### 6.5 Recompense

Pour chaque tache, cout = latence / 1 s + 2 × (deadline manquee) + 0.3 ×
energie locale, recompense = −cout (bornee a −15).

- Device : energie de calcul ou d'emission du device, ponderee par
  1 + 2 × (1 − batterie) (plus chere quand la batterie est faible).
- Edge : energie de calcul et d'emission (vers le cloud ou l'edge voisin) de l'edge.
- Agent centralise : somme des deux.

La latence de bout en bout est partagee par tous les niveaux, ce qui pousse a
une coordination implicite sans echange d'etat complet.

## 7. Entrainement

Entrainement sequentiel, pas simultane (evite l'instabilite du MARL simultane) :

1. round 0 : agent edge contre un device heuristique (80 % des taches vers
   l'edge), puis agent device contre l'agent edge fige ;
2. round 1 : on recommence avec les agents du round 0 comme partenaires figes.

Budget : 100 000 pas pour l'edge et 150 000 pour le device, par round. Charge
tiree au hasard dans [0.5, 4] a chaque episode.

**Baseline RL centralisee (`flat_rl`)** : un seul agent DQN avec vue globale
exacte (10 valeurs : tache, batterie, toutes les files, backlog du cloud) et
4 actions (device, edge, cloud, edge voisin), meme budget total (500 000 pas).

Hyperparametres DQN (identiques pour tous) : MLP 64×64, lr 1e-3, buffer 50 000,
batch 64, gamma 0.9, epsilon 1 → 0.05 sur 40 % des pas, mise a jour de la
cible tous les 500 pas.

## 8. Baselines

- `random` : niveau tire uniformement (device, edge, cloud).
- `always_edge` : tout vers l'edge.
- `greedy_nearest` : device si sa file locale tient la deadline, sinon edge si
  sa file tient la deadline, sinon cloud. Chaque niveau n'utilise que sa propre
  information, comme la hierarchie.
- `edf_like` : choisit le niveau qui termine le plus tot (device, edge, edge
  voisin ou cloud, la fin au cloud etant celle de l'ordonnancement EDF), avec la
  vue globale exacte de toutes les files. C'est une reference clairvoyante, pas
  une heuristique deployable.
- `flat_rl` : RL centralise (section 7).

Toutes les politiques partagent le meme cloud EDF : seule la decision de
placement change. Elles sont evaluees sur les memes flux de taches.

## 9. Metriques

- **Latence** : latence moyenne de bout en bout (s).
- **Deadlines tenues** : part des taches finies avant leur deadline.
- **Energie** : energie moyenne par tache (device + edge, en J).
- **Repartition** : part des taches executees sur le device, l'edge et le cloud ;
  **transfert** : part envoyee a l'edge voisin ; **refus** : part refusee par le cloud.
- **Surcout de controle** : valeurs scalaires de controle echangees par tache
  (hors donnees). Vue globale : 10 valeurs d'etat + 1 decision par tache.
  Hierarchie : beacons periodiques + 4 valeurs par descripteur de tache
  transmis a l'edge, au cloud ou a l'edge voisin.

Evaluation : 20 episodes par charge (0.5, 1, 2, 3, 4), graines jamais vues a
l'entrainement, moyennes avec IC95.

## 10. Resultats (trace Alibaba)

Tableaux complets : `results/summary.md` ; donnees : `results/results.csv`.

Charge ×3 :

| Politique | Latence (s) | Deadlines tenues | Energie (J/tache) | Controle (valeurs/tache) |
|---|---|---|---|---|
| Aleatoire | 3.97 | 71.6 % | 0.90 | 0 |
| Toujours edge | 9.86 | 25.4 % | 0.49 | 0 |
| Greedy niveau le plus proche | 1.17 | 97.6 % | 0.85 | 4.1 |
| EDF-like (vue globale) | 0.73 | 96.8 % | 0.37 | 11.0 |
| RL centralise (flat) | 0.90 | 95.5 % | 0.40 | 11.0 |
| **RL hierarchique** | 0.89 | 96.0 % | **0.31** | 6.5 |

Constats :

- Le RL hierarchique fait aussi bien que le RL centralise en latence et en
  deadlines a toutes les charges, avec l'energie la plus basse de toutes les
  politiques, et 10 a 47 % de valeurs de controle en moins des la charge ×1.
- Face au greedy : latence reduite d'environ 25 % et energie divisee par
  environ 2.7, pour un taux de deadlines legerement plus bas (le greedy calcule
  beaucoup en local et consomme la batterie).
- EDF-like (vue globale exacte) reste le plus rapide : c'est une borne, pas une
  methode deployable.
- L'agent edge utilise beaucoup l'edge voisin (26 a 50 % des taches), le lien
  edge-edge etant peu couteux.
- A tres faible charge (×0.5), la hierarchie echange plus de valeurs que la vue
  globale (13.3 contre 11.0) : les beacons et descripteurs coutent plus que ce
  qu'ils remplacent quand il y a peu de taches.

**Periode des beacons** (RL hierarchique, charge ×3, `results/beacon_study.csv`) :

| Periode (s) | 0.25 | 0.5 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|
| Latence (s) | 0.90 | 0.89 | 0.89 | 0.91 | 1.01 | 1.07 |
| Deadlines tenues (%) | 96.3 | 96.1 | 96.0 | 94.8 | 91.8 | 90.5 |
| Controle (valeurs/tache) | 9.7 | 7.6 | 6.5 | 6.0 | 5.9 | 5.9 |

Des beacons plus frequents coutent plus cher sans gain au-dela de 1 s ; plus
rares, l'information devient trop perimee et les deadlines chutent.

**Figures (`results/`)** :

| Figure | Contenu |
|---|---|
| `fig_latency.png` | Latence moyenne selon la charge (echelle log) |
| `fig_hit_rate.png` | Deadlines tenues selon la charge |
| `fig_energy.png` | Energie par tache selon la charge |
| `fig_overhead.png` | Surcout de controle contre latence, charge ×3 |
| `fig_beacon.png` | Effet de la periode des beacons |
| `fig_training.png` | Courbes d'apprentissage des agents |

## 11. Hypotheses et limites (a ecrire dans l'article)

- EDF au cloud seulement ; devices et edges restent en FIFO.
- Avec EDF, une tache cloud acceptee peut etre doublee plus tard par une tache
  plus urgente. La recompense RL utilise la fin estimee au moment de la
  decision ; les metriques rapportees utilisent la fin reelle.
- `edf_like` est une interpretation de la baseline EDF du sujet (placement au
  plus tot fini, vue globale, cloud EDF). A confirmer avec l'encadrante.
- Trace : sous-ensemble Alibaba (20 000 taches `batch_task` terminees, debut de
  la trace). Seuls les instants de debut et les durees sont utilises ; taille et
  deadline sont derivees. La trace a une resolution d'une seconde, d'ou de
  nombreuses arrivees simultanees (rafales) et des IC95 assez larges.
- Le retour du resultat est neglige. L'energie du cloud n'est pas comptee. Le
  fading est tire par tache.
- Un seul entrainement (une graine) par agent : les ecarts entre RL
  hierarchique et RL centralise doivent etre lus avec prudence. Pour consolider,
  relancer `train.py --seed 1` et comparer.
- Resultats obtenus sur ce simulateur simplifie, pas sur un reseau reel.

## 12. Extension (non faite)

Variante federee de la hierarchie : chaque edge agregerait les mises a jour de
politique de ses devices (federation device → edge), puis le cloud agregerait
les politiques des edges (federation edge → cloud), sans partage des donnees
brutes d'usage.

## 13. References

- PureEdgeSim, EdgeCloudSim, iFogSim (simulateurs edge/cloud)
- Hierarchical Reinforcement Learning for Task Offloading in MEC (IEEE)
- Federated Reinforcement Learning, survey (arXiv)
- ETSI Multi-access Edge Computing (MEC), specifications
- stable-baselines3
- Alibaba cluster-trace-v2018 : https://github.com/alibaba/clusterdata
