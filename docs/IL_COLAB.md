# Entraîner ACT sur Google Colab (GPU cloud)

Objectif : entraîner le modèle sur un GPU cloud (T4 gratuit / A100 avec Colab Pro), plus
rapide qu'un entraînement local sur CPU/MPS. Référence : notebook officiel LeRobot
https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb

Principe : le dataset est poussé sur le Hub HF (dépôt privé), Colab le télécharge, entraîne,
puis repousse le modèle sur le Hub ; le modèle est ensuite rapatrié pour l'évaluation locale.

Remplacer partout `<USER>` par le nom d'utilisateur Hugging Face réel.

---

## Prérequis (une fois)
1. Créer un compte gratuit sur https://huggingface.co et noter le username (`<USER>`).
2. Token : Settings → Access Tokens → New token → type Write (ou cocher toutes les cases
   "Repositories"), puis le copier.
3. En local : `hf auth login`, puis coller le token.

## Étape 1 — Pousser le dataset sur le Hub (local, ~1,5 Go)
Créer d'abord le dépôt en privé sur le site (New → Dataset, nom `so101_orange_cube`,
Private), puis envoyer le contenu local :
```bash
hf upload <USER>/so101_orange_cube \
  ~/.cache/huggingface/lerobot/<USER>/so101_orange_cube \
  --repo-type=dataset
```
> 1,5 Go représentent environ 10-30 min selon la connexion. Le nom du dossier local
> (`<USER>/...`) n'a pas d'importance ; seul compte le dépôt Hub `<USER>/so101_orange_cube`.

## Étape 2 — Le notebook Colab
1. Ouvrir le notebook officiel (lien plus haut).
2. Runtime → Change runtime type → GPU (T4 = gratuit ; A100/L4 = Colab Pro, plus rapide).
3. Exécuter les cellules dans l'ordre. Elles :
   - installent LeRobot (`git clone` + `pip install -e ".[train,dataset]"` + ffmpeg),
   - lancent `hf auth login` (coller le token pour télécharger le dataset privé),
   - lancent l'entraînement.
4. Remplacer la cellule d'entraînement par (en adaptant `<USER>`) :
   ```bash
   !lerobot-train \
     --dataset.repo_id=<USER>/so101_orange_cube \
     --policy.type=act \
     --output_dir=outputs/train/act_so101 \
     --job_name=act_so101 \
     --policy.device=cuda \
     --wandb.enable=False \
     --policy.repo_id=<USER>/act_so101_orange_cube \
     --batch_size=16 \
     --steps=50000
   ```
   - `--policy.repo_id` = destination Hub du modèle entraîné (poussé à la fin).
   - T4 gratuit : `--batch_size=16 --steps=50000` (quelques heures, tient dans une session).
   - A100 (Pro) : `--batch_size=64 --steps=100000` (~1,5 h).

> Colab gratuit peut se déconnecter : garder l'onglet actif. Seul le modèle final
> (`--policy.repo_id`) est poussé sur le Hub ; si la session se termine avant la fin, le
> travail est perdu (VM effacée). D'où des `--steps` modérés sur T4 (50k), ou Colab Pro
> pour 100k.

## Étape 3 — Rapatrier le modèle pour l'évaluation locale
Le modèle est sur `<USER>/act_so101_orange_cube`. Pour évaluer, pointer l'évaluation
directement dessus (LeRobot télécharge le modèle automatiquement) :
```bash
python scripts/eval_policy.py --policy-path <USER>/act_so101_orange_cube --episodes 1
```
(évaluer un épisode à la fois, en ramenant le bras à la position home entre chaque essai)

---

## Notes
- Le clone frais de LeRobot sur Colab (environnement neuf) n'a pas le bug groot/transformers
  corrigé localement ; aucun patch n'est nécessaire côté Colab.
- Colab sert uniquement à accélérer l'entraînement. Le dataset reste dans le compte HF
  (dépôt privé) ; la VM Colab est effacée après la session.
- `~100k` steps constituent la cible (au-delà, risque de sur-apprentissage). Pour améliorer
  au-delà de ce seuil, ajouter des démonstrations plutôt que des steps.
