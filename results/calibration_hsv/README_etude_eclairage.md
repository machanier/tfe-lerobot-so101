# Étude d'éclairage — choix de la lumière pour la calibration et les tests

But : comparer plusieurs conditions d'éclairage, en capturant la **même scène**
depuis les **3 caméras**, pour décider laquelle garder. La condition retenue
servira ensuite pour :
1. la calibration HSV (`scripts/calibrate_hsv.py`),
2. tous les tests de perception/saisie,
3. la campagne finale.

> Ce dossier est sous `outputs/` → **gitignoré** (les images ne sont pas
> commitées, c'est voulu). La *décision* finale, elle, doit être consignée en
> texte versionné — voir la section « Décision » en bas + le mémoire
> (annexe calibration / décisions Dxx).

---

## Principe : ne fais varier QUE la lumière

Pour que la comparaison soit honnête :
- **scène figée** : mêmes objets, mêmes positions, même cadrage des 3 caméras ;
- on ne change **que l'éclairage** d'une condition à l'autre ;
- une condition = un sous-dossier numéroté ici.

Pourquoi viser un éclairage **artificiel constant** (volets fermés) plutôt que
la lumière du jour : le jour dérive (heure, nuages, soleil) ; un éclairage fixe
est reproductible à 10h comme à 22h. C'est la condition de reproductibilité
attendue par le mémoire (sensibilité à l'éclairage du seuillage HSV).

---

## Protocole de capture

Pour chaque condition, capture les 3 caméras dans le **même** sous-dossier.
`preview_camera.py` crée le dossier au besoin ; touche `s` = sauver l'image
courante, `q` = quitter.

```bash
# Exemple — condition "01_plafond_seul"
python scripts/preview_camera.py --camera 0 --output-dir outputs/lighting_study/01_plafond_seul
python scripts/preview_camera.py --camera 1 --output-dir outputs/lighting_study/01_plafond_seul
python scripts/preview_camera.py --camera 2 --output-dir outputs/lighting_study/01_plafond_seul
```

(Les fichiers sont nommés `preview_cam_<idx>_<timestamp>.png`, donc cam_0/1/2
restent distinctes.)

Convention de nommage des conditions (à adapter) :
```
01_plafond_seul/
02_plafond_plus_lampe_bureau/
03_deux_lampes_diffuses/
04_volets_ouverts_jour/         # pour comparaison, mais non reproductible
...
```

---

## Quoi regarder en comparant

Pour chaque condition, juge :
- **Reflets spéculaires** sur les objets → mauvais (pixels délavés vers le blanc,
  faussent S/V). Lumière diffuse > lumière directe.
- **Surexposition** : couleurs délavées, blanc « cramé » → V au plafond, S chute.
- **Sous-exposition** : image sombre, grain/bruit → V trop bas, faux blobs.
- **Uniformité** : la lumière tombe-t-elle de façon homogène sur toute la table ?
- **Lisibilité des couleurs** : les teintes sont-elles franches et saturées ?

La bonne condition = **lumineuse mais pas cramée**, **diffuse**, **homogène**,
sans reflets, avec des couleurs franches.

---

## Décision (à remplir, puis à reporter dans le mémoire)

| Condition | Description (lampes, volets, heure) | Reflets | Expo (sombre/OK/cramé) | Uniformité | Retenu ? |
|-----------|-------------------------------------|---------|------------------------|------------|----------|
| 01_...    |                                     |         |                        |            |          |
| 02_...    |                                     |         |                        |            |          |
| 03_...    |                                     |         |                        |            |          |

**Condition retenue : ______**

**Réglage précis à reproduire** (positions/intensités des lampes, volets, etc.,
pour pouvoir recréer exactement cet éclairage à chaque session) :

> _(décris ici le montage exact)_

Une fois la décision prise, lance la calibration **sous cette condition exacte** :
```bash
python scripts/calibrate_hsv.py --camera 0
```
et n'en change plus jusqu'à la fin de la campagne.
