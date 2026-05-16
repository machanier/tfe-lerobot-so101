# Makefile pour la compilation du memoire TFE Maxence Chanier
# Usage :
#   make          (alias pour 'make pdf')
#   make pdf      (compile main.pdf via latexmk)
#   make clean    (efface les fichiers intermediaires)
#   make watch    (recompile automatiquement a chaque sauvegarde)
#   make wc       (compte de mots approximatif via texcount)

MAIN = main
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -file-line-error -synctex=1

.PHONY: all pdf clean watch wc help

all: pdf

pdf:
	latexmk $(LATEXMK_FLAGS) $(MAIN).tex

watch:
	latexmk $(LATEXMK_FLAGS) -pvc $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.toc *.tdo *.lot *.lof *.synctex.gz
	rm -f chapters/*.aux

wc:
	@which texcount > /dev/null 2>&1 && texcount -inc -total -sum $(MAIN).tex || \
		echo "texcount non installe. Installer via 'tlmgr install texcount'."

help:
	@echo "Commandes disponibles :"
	@echo "  make          - compile main.pdf"
	@echo "  make pdf      - idem"
	@echo "  make watch    - compile en continu (latexmk -pvc)"
	@echo "  make clean    - efface les fichiers intermediaires + le PDF"
	@echo "  make wc       - compte de mots (necessite texcount)"
