#!/bin/bash
# ============================================================
# setup_env.sh – Mise en place de l'environnement de travail
# ============================================================
# Usage : ./setup_env.sh
# ============================================================

set -e

PYTHON_VERSION="3.12"
VENV_DIR="venv"
LEROBOT_DIR="lerobot"

echo "======================================"
echo "  Setup TFE LeRobot SO-101"
echo "======================================"

# --- 1. Vérifier Python ---
echo ""
echo "▶ Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installe-le avec : brew install python@${PYTHON_VERSION}"
    exit 1
fi
echo "  ✅ $(python3 --version)"

# --- 2. Créer le venv ---
echo ""
echo "▶ Création de l'environnement virtuel..."
if [ -d "$VENV_DIR" ]; then
    echo "  ⚠️  Le dossier $VENV_DIR existe déjà. Supprime-le si tu veux repartir de zéro."
else
    python3 -m venv "$VENV_DIR"
    echo "  ✅ venv créé dans ./$VENV_DIR"
fi

# --- 3. Activer le venv ---
echo ""
echo "▶ Activation du venv..."
source "$VENV_DIR/bin/activate"
echo "  ✅ venv activé ($(which python))"

# --- 4. Mettre à jour pip ---
echo ""
echo "▶ Mise à jour de pip..."
pip install --upgrade pip setuptools wheel

# --- 5. Cloner et installer LeRobot ---
echo ""
echo "▶ Installation de LeRobot..."
if [ -d "$LEROBOT_DIR" ]; then
    echo "  ⚠️  Le dossier $LEROBOT_DIR existe déjà, mise à jour..."
    cd "$LEROBOT_DIR"
    git pull
    cd ..
else
    echo "  Clonage du repo LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

echo "  Installation en mode éditable avec support Feetech..."
pip install -e "./$LEROBOT_DIR[feetech]"

# --- 6. Dépendances supplémentaires ---
echo ""
echo "▶ Installation des dépendances supplémentaires..."
pip install -r requirements.txt

# --- 7. Résumé ---
echo ""
echo "======================================"
echo "  ✅ Setup terminé !"
echo "======================================"
echo ""
echo "  Pour activer le venv :"
echo "    source venv/bin/activate"
echo ""
echo "  Pour vérifier :"
echo "    python -c 'import lerobot; print(lerobot.__version__)'"
echo ""
echo "  Pour téléopérer :"
echo "    python scripts/teleoperate.py"
echo ""
