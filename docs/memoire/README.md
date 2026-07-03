# Mémoire TFE — Maxence Chanier

Brouillon LaTeX du mémoire de Travail de Fin d'Études.

## Compilation

```bash
make            # compile main.pdf
make watch      # compile en continu (utile pendant la rédaction)
make clean      # efface fichiers intermédiaires + PDF
```

Tu as déjà MacTeX installé (`pdflatex`, `biber`, `latexmk` détectés).
Aucune installation supplémentaire requise.

## Structure

```
docs/memoire/
├── main.tex                          point d'entrée
├── biblio.bib                        bibliographie (sync depuis Zotero)
├── Makefile                          commandes de compilation
├── README.md                         ce fichier
├── chapters/
│   ├── 00_titlepage.tex              page de titre
│   ├── 00_abstract.tex               résumé (à rédiger)
│   ├── 00_remerciements.tex          (à rédiger)
│   ├── 01_introduction.tex           ✅ pré-rempli depuis cahier des charges
│   ├── 02_etat_de_lart.tex           ✅ pré-rempli depuis biblio + décisions
│   ├── 03_architecture.tex           ✅ pré-rempli depuis PROJECT_STATUS
│   ├── 04_perception.tex             ✅ pré-rempli (HSV + OWL-ViTv2 + D6-D13)
│   ├── 05_planning.tex               ✅ pré-rempli (grasp top-down)
│   ├── 06_control_evaluation.tex     🟡 stub structuré (à remplir Sprint 3-5)
│   ├── 07_discussion.tex             🟡 stub structuré
│   ├── 08_conclusion.tex             🟡 stub structuré
│   └── A1_calibration_details.tex    🟡 stub annexe
└── figures/                          dossier pour les figures (vide pour l'instant)
```

## Légende

- ✅ Pré-rempli avec contenu substantiel.
- 🟡 Stub structuré : sections définies, contenu à rédiger.
- ⬜ À faire de zéro.

## Liste des TODO

Tous les emplacements à compléter sont marqués avec `\todoF{...}` (TODO
détaillés en rouge) ou `\todo{...}` (TODO courts en marge). Une liste des
TODO est automatiquement générée en début de document (après la table des
matières).

## Pour rédiger

1. `make watch` (lance la recompilation automatique).
2. Ouvre `main.pdf` dans un PDF viewer qui se met à jour automatiquement
   (Preview sur Mac est suffisant).
3. Ouvre `chapters/XX_*.tex` dans ton éditeur.
4. Modifie, sauvegarde, le PDF se met à jour automatiquement.

## Bibliographie

Le fichier `biblio.bib` est une copie de `docs/references/tfe_zotero.bib/tfe_zotero.bib.bib`.
Pour mettre à jour : copier le nouveau .bib depuis Zotero, OU éditer
directement ce fichier.

Pour citer une référence : `\cite{cle_bibtex}`. La clé est le premier mot
après `@article{` ou `@misc{` dans le .bib.

## Compteur de mots / pages

```bash
make wc           # compte de mots approximatif (nécessite texcount)
```

Un mémoire bachelor UNIGE en informatique fait typiquement 30-50 pages
(hors annexes), soit ~10000-15000 mots.
