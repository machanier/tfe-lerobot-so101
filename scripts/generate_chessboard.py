#!/usr/bin/env python3
"""
generate_chessboard.py - Genere un damier de calibration imprimable en PNG.

Le PNG est genere a haute resolution (300 DPI par defaut) avec la metadonnee
DPI embarquee, pour que l'imprimante respecte les dimensions physiques quand
on imprime a 100 % (sans mise a l'echelle).

Pourquoi PNG plutot que SVG : les imprimantes/visionneuses interpretent les
SVG de facon parfois aleatoire ; le PNG en haute resolution est plus
predictible. Pour verifier que l'impression est correcte, mesurer au pied a
coulisse un carre apres impression.

Convention damier : asymetrique (cols != rows), pour eviter l'ambiguite a 4
plis de la detection OpenCV sur un damier carre. Recommande : 9 colonnes x 6
lignes de coins internes, comme la plupart des tutoriels de calibration.

Usage :
    python scripts/generate_chessboard.py
    python scripts/generate_chessboard.py --cols 9 --rows 6 --square-mm 22
    python scripts/generate_chessboard.py --cols 7 --rows 5 --square-mm 25 --dpi 600
"""

import argparse
import os

from PIL import Image, ImageDraw

MM_PER_INCH = 25.4


def generate_png(rows_inner, cols_inner, square_mm, margin_mm, dpi, output_path):
    """Genere un PNG du damier avec DPI embarque."""
    px_per_mm = dpi / MM_PER_INCH
    sq_px = int(round(square_mm * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    rows_squares = rows_inner + 1
    cols_squares = cols_inner + 1
    board_w_px = cols_squares * sq_px
    board_h_px = rows_squares * sq_px
    total_w_px = board_w_px + 2 * margin_px
    total_h_px = board_h_px + 2 * margin_px

    img = Image.new("L", (total_w_px, total_h_px), color=255)  # fond blanc
    draw = ImageDraw.Draw(img)

    for r in range(rows_squares):
        for c in range(cols_squares):
            if (r + c) % 2 == 0:
                x0 = margin_px + c * sq_px
                y0 = margin_px + r * sq_px
                # -1 sur x1/y1 pour eviter le chevauchement entre carres voisins
                draw.rectangle([x0, y0, x0 + sq_px - 1, y0 + sq_px - 1], fill=0)

    img.save(output_path, dpi=(dpi, dpi))

    # Tailles reelles a l'impression (en mm), retour pour info
    actual_w_mm = total_w_px / px_per_mm
    actual_h_mm = total_h_px / px_per_mm
    actual_sq_mm = sq_px / px_per_mm
    return actual_w_mm, actual_h_mm, actual_sq_mm


def main():
    parser = argparse.ArgumentParser(description="Genere un damier de calibration en PNG.")
    parser.add_argument("--cols", type=int, default=9,
                        help="Coins internes en largeur (defaut: 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="Coins internes en hauteur (defaut: 6)")
    parser.add_argument("--square-mm", type=float, default=22.0,
                        help="Taille d'un carre en mm (defaut: 22)")
    parser.add_argument("--margin-mm", type=float, default=10.0,
                        help="Marge blanche autour du damier (defaut: 10 mm)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution d'impression (defaut: 300)")
    parser.add_argument("--output-dir", type=str, default="outputs/chessboards")
    args = parser.parse_args()

    if args.cols == args.rows:
        print(f"ATTENTION : damier symetrique {args.cols}x{args.rows}. OpenCV a une")
        print(f"ambiguite de detection a 4 plis dans ce cas. Utilise une dimension")
        print(f"paire et l'autre impaire (ex: 9x6) pour eviter ce probleme.")
        print()

    os.makedirs(args.output_dir, exist_ok=True)
    name = f"chessboard_{args.cols}x{args.rows}_{args.square_mm:g}mm_{args.dpi}dpi.png"
    png_path = os.path.join(args.output_dir, name)

    w_mm, h_mm, sq_mm = generate_png(
        args.rows, args.cols, args.square_mm, args.margin_mm, args.dpi, png_path
    )

    print(f"Damier genere : {png_path}")
    print(f"  Coins internes : {args.cols} (cols) x {args.rows} (rows)")
    print(f"  Carres         : {args.cols + 1} x {args.rows + 1} carres")
    print(f"  Taille carre   : {args.square_mm:g} mm "
          f"(reel apres rasterisation : {sq_mm:.4f} mm)")
    print(f"  Damier complet : "
          f"{(args.cols + 1) * args.square_mm:.1f} x {(args.rows + 1) * args.square_mm:.1f} mm")
    print(f"  Feuille (avec marges) : {w_mm:.1f} x {h_mm:.1f} mm")
    print(f"  Resolution     : {args.dpi} DPI")
    print()
    print("Impression :")
    print("  1. Ouvrir le PNG dans Preview (macOS).")
    print("  2. Fichier > Imprimer (Cmd+P).")
    print("  3. Bien decocher 'Echelle pour adapter a la page' /")
    print("     'Scale to fit'. Choisir taille reelle / 100 %.")
    print("  4. Apres impression, MESURER un carre au pied a coulisse :")
    print(f"     il doit faire {args.square_mm:g} mm a moins de 0.1 mm pres.")
    print("  5. Si la mesure est differente, la passer en --square-size lors")
    print("     de la calibration extrinseque (la calibration intrinseque a")
    print("     deja ete faite avec ton damier 7x7 actuel, elle reste valide).")


if __name__ == "__main__":
    main()
