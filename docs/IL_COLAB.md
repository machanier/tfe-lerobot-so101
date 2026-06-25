# Entraîner ACT sur Google Colab (GPU rapide)

But : entraîner le modèle sur un vrai GPU cloud (T4 gratuit / A100 si Pro) au lieu du M4
(~1,14 s/step). Réf : notebook officiel LeRobot
https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb

Principe : dataset poussé sur le Hub HF (privé) -> Colab le télécharge, entraîne,
repousse le **modèle** sur le Hub -> on le rapatrie pour l'éval locale.

Remplace partout `<USER>` par ton **vrai pseudo Hugging Face**.

---

## Prérequis (une fois)
1. Crée un compte gratuit sur https://huggingface.co → note ton **username** (`<USER>`).
2. Token : Settings → Access Tokens → **New token** → type **Write** (ou coche toutes les
   cases "Repositories"). Copie-le.
3. En local : `hf auth login` puis colle le token.

## Étape 1 — Pousser le dataset sur le Hub (local, ~1,5 Go)
Crée d'abord le dépôt en **privé** sur le site (New → **Dataset**, nom `so101_orange_cube`,
Private), puis envoie le contenu local :
```bash
hf upload <USER>/so101_orange_cube \
  ~/.cache/huggingface/lerobot/maxence/so101_orange_cube \
  --repo-type=dataset
```
> 1,5 Go → compte ~10-30 min selon ta connexion. (Le nom local reste "maxence/...", peu
> importe : ce qui compte c'est le dépôt Hub `<USER>/so101_orange_cube`.)

## Étape 2 — Le notebook Colab
1. Ouvre le notebook officiel (lien plus haut).
2. **Runtime → Change runtime type → GPU** (T4 = gratuit ; A100/L4 = Colab Pro, plus rapide).
3. Exécute les cellules dans l'ordre. Elles font :
   - installer LeRobot (`git clone` + `pip install -e ".[train,dataset]"` + ffmpeg),
   - `hf auth login` → **colle ton token** (pour télécharger ton dataset privé),
   - lancer l'entraînement.
4. **Remplace la cellule d'entraînement** par (adapte `<USER>`) :
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
   - `--policy.repo_id` = où le **modèle entraîné** sera poussé sur le Hub (à la fin).
   - **T4 gratuit** : `--batch_size=16 --steps=50000` (~quelques heures, tient dans une session).
   - **A100 (Pro)** : `--batch_size=64 --steps=100000` (~1,5 h).

> ⚠️ Colab gratuit peut se déconnecter : **garde l'onglet actif**. Seul le modèle **final**
> (`--policy.repo_id`) est poussé sur le Hub ; si la session meurt avant la fin, le travail
> est perdu (VM effacée). D'où des `--steps` raisonnables sur T4 (50k), ou Colab Pro pour 100k.

## Étape 3 — Rapatrier le modèle pour l'éval locale
Le modèle est sur `<USER>/act_so101_orange_cube`. Pour évaluer, pointe l'éval directement
dessus (LeRobot télécharge le modèle tout seul) :
```bash
python scripts/eval_policy.py --policy-path <USER>/act_so101_orange_cube --episodes 1
```
(rappel : éval **une à une**, et tu remets le bras à home entre chaque essai)

---

## Notes
- Le clone frais de LeRobot sur Colab n'a PAS le bug groot/transformers qu'on a patché en
  local (environnement neuf) → rien à reporter là-bas.
- Mémoire/TFE : Colab sert juste à entraîner vite. Le dataset reste dans **ton** compte HF
  (dépôt privé) ; la VM Colab est effacée après la session.
- `~100k` steps = cible (au-delà = sur-apprentissage). Ce qui aide au-delà = plus de démos,
  pas plus de steps.
