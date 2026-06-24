# Retours de Maxence — par partie du mémoire

> **Mode d'emploi.** Ce fichier suit la table des matières du mémoire. Quand tu as
> une remarque, descends jusqu'à la partie concernée et écris-la **sous le titre**.
> Tu n'as donc pas à recopier la référence : l'endroit où tu écris EST la référence.
>
> Pour pointer une phrase précise, ajoute quelques mots **entre guillemets « … »**
> (4–10 mots suffisent, je les retrouve dans le source).
> Optionnel : commence ta ligne par un tag — **[Q]** question · **[M]** modif · **[V]** à vérifier.
>
> Exemple (sous 6.2) :
>   [Q] « pondération translation/rotation » — pourquoi 0,1 ? ajouter une phrase d'explication.
>   [M] trop personnel à mon goût, reformuler plus sobrement.
>
> Quand tu as fini (ou par lots), dis-moi **« lis mes retours »** : je lis ce fichier,
> je réponds à tes questions et j'applique les modifs.

---

## Résumé
1. "reste souvent dispersée ou enfouie dans des solutions propriétaires." -> comment cela ? Que cela signifie et comment le sais tu ?
2. "politique apprise de bout en bout" -> cela fait référence à mon bonus et à ce qui se fait de base avec ce genre de robot qui de passer par de l'imitation learning ?
3. "entièrement open-source et reproductible" -> de préciser cela c'est parce que mon repo github est public ? C'est quoi ? Il faudra que je mette ce travail public de toute manière ? Je veux simplement comprendre
4. "de la mettre en perspective avec le paradigme end-to-end
de l’apprentissage par imitation, dans le cadre du cahier des charges, qui vise à terme la saisie en
environnement encombré" -> du coup cela fait le lien avec la remarque 2 ?
5. "(méthode de Zhang)" -> fait référence à la biblio <3D Hand-Eye Calibration for Collaborative Robot  Arm: Look at Robot Base Once> ?
6. DLT, URDF, Gauss–Newton amorti par Levenberg–Marquardt, quintique -> des mots que je ne comprends pas
7. Mots-clés -> est une manière de faire dans un mémoire, c'est voulu quoi ?


8. Des répétitions successives

## Remerciements
1. première phrase -> Le sujet vient de moi et moi seul. Pour la petite histoire, y a un an j'ai voulu mettre en place ce projet que j'ai découvert sur youtube, je n'avais aucune connaissance dans le domaine, je suis tout de même en bachelor d'informatique, mais avec les cours cela était compliqué. De là, en voyant la fin de l'année passé, les collègues commencait deja a chercher un sujet pour le projet de bachelor de l anne suivante, et moi ayant cette idée en tête je me suis dis que je pourrais trouvé quelque chose, un sujet, dans lequel je pourrais intégrer ce robot. J'ai donc commencé l'été par imprimer et construire le robot en entier et c'est tout, j'en étais là. Fin de l'année il fallait trouvé un encadrant, un professeur, qui puisse nous suivre durant le projet et aussi en premier voir avec lui pour trouver un sujet. Moi ayant apporté directement l'idée de ce robot et ayant chercher quoi faire un peu autour de ce robot, on a discuté à deux reprises sur le sujet avant de lui apporté le sujet définitif qu'il a validé. Il a donc essayé de comprendre pendant nos 2 discussion comment je pouvais, trouvait cela même compliqué mais avait énormément confiance en moi car je me trouvais très motivé. 




# 1. Introduction
## 1.1 Contexte

## 1.2 Problématique

## 1.3 Objectifs

## 1.4 Matériel
1. CAO faite via Fusion 360
2. Montrer la décision prise pour les mesures de la structure 3d qui a été mis en archive dans le dossier je crois
3. Pas compris ceci "soit deux contrôleurs xHCI distincts, faute de quoi la troisième caméra n’est pas énumérée."

## 1.5 Plan du mémoire


# 2. État de l'art
## 2.1 Calibration caméra-robot (hand-eye)

## 2.2 Triangulation stéréo et géométrie projective

## 2.3 Estimation de pose monoculaire (PnP)

## 2.4 Détection d'objets : du seuillage classique aux foundation models
### 2.4.1 Détection par seuillage couleur (classique)

### 2.4.2 Détection apprise (closed-vocabulary)

### 2.4.3 Détection open-vocabulary

## 2.5 Grasp planning

## 2.6 Planification de trajectoires

## 2.7 Imitation learning et politiques apprises

## 2.8 Positionnement de ce travail


# 3. Architecture
## 3.1 Vue d'ensemble

## 3.2 Interfaces stables (dataclasses)

## 3.3 Interfaces abstraites (Strategy pattern)

## 3.4 Repères et conventions
### 3.4.1 Repère base du robot

### 3.4.2 Convention T_A^B

## 3.5 Lien hardware–logiciel

## 3.6 Mode développement live vs replay

## 3.7 Gestion des dépendances


# 4. Perception
## 4.1 Pipeline général

## 4.2 Calibration et modèles caméra
### 4.2.1 Calibration intrinsèque

### 4.2.2 Calibration extrinsèque (hand-eye)

## 4.3 Détection 2D : version V1 (HSV)
### 4.3.1 Principe

### 4.3.2 Calibration des plages HSV

### 4.3.3 Limites identifiées du V1

## 4.4 Détection 2D : version V2 (open-vocabulary)
### 4.4.1 Choix du modèle : OWL-ViTv2

### 4.4.2 Descriptions textuelles enrichies

### 4.4.3 Limites identifiées du V2

## 4.5 Reconstruction 3D : triangulation stéréo et PnP
### 4.5.1 Triangulation stéréo (DLT)

### 4.5.2 Sensibilité de la profondeur à l'erreur de détection

### 4.5.3 PnP monoculaire (fallback)

### 4.5.4 Estimation de la géométrie de l'objet

### 4.5.5 Filtres post-estimation

## 4.6 Synchronisation multi-caméras et architecture USB

## 4.7 Validation expérimentale


# 5. Planification de saisie
## 5.1 Définition du problème

## 5.2 Stratégie retenue : top-down grasp
### 5.2.1 Pourquoi le top-down, et pas autre chose

### 5.2.2 Construction géométrique

### 5.2.3 Rotation R(θ) : pince verticale alignée

### 5.2.4 Orientation wrist_roll : mâchoires en travers du petit axe

### 5.2.5 Ouverture de pince adaptative

### 5.2.6 Décalage latéral vers le doigt fixe

### 5.2.7 Profondeur de prise ancrée sur la table

### 5.2.8 Découverte : la convention de fermeture à 90°

### 5.2.9 Filtres et garde-fous

## 5.3 Au-delà du top-down : la prise latérale

## 5.4 Interface avec le module de contrôle

## 5.5 Validation et tests


# 6. Contrôle et évaluation expérimentale
## 6.1 Cinématique directe (forward kinematics)

## 6.2 Cinématique inverse (inverse kinematics)

## 6.3 Génération de trajectoire

## 6.4 Module de contrôle moteur

## 6.5 Pipeline intégré

## 6.6 Évaluation expérimentale
### 6.6.1 Protocole de la campagne formelle

### 6.6.2 Résultats par configuration

### 6.6.3 Comparaison pipeline modulaire vs imitation learning


# 7. Discussion
## 7.1 Synthèse des limites identifiées
### 7.1.1 Limites de la perception V1 (HSV)

### 7.1.2 Limites de la perception V2 (OWL-ViTv2)

### 7.1.3 Limites du grasp planner (top-down)

### 7.1.4 Limites de la précision géométrique

## 7.2 Choix académiques discutables et résultat sur la sous-actionation
### 7.2.1 Pourquoi OWL-ViTv2 plutôt que Grounding-DINO ?

### 7.2.2 Pourquoi pas un détecteur fine-tuné (YOLO) ?

### 7.2.3 Pourquoi 5 DDL, et ce que cela implique

### 7.2.4 La convention de fermeture à 90° : un débogage révélateur

## 7.3 Comparaison des deux paradigmes (modulaire vs imitation)

## 7.4 Pistes d'amélioration


# 8. Conclusion
## 8.1 Bilan du travail réalisé

## 8.2 Contributions

## 8.3 Perspectives

## 8.4 Remarque finale sur l'utilisation des outils IA


# Annexe A — Détails de calibration
## A.1 Structure du code

## A.2 Repère base : mesure expérimentale

## A.3 Décision D1 : recalage du wrist_roll

## A.4 Décision D2 : damier asymétrique 9×6

## A.5 Décision D3 : solveur hand-eye robuste

## A.6 Procédure de capture des données extrinsèques
