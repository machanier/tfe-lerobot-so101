#!/usr/bin/env python3
"""
generate_chessboard.py – Genere un damier de calibration au format SVG.

Sortie : SVG vectoriel, pret a imprimer (impression a 100% obligatoire).

Usage :
    python scripts/generate_chessboard.py
    python scripts/generate_chessboard.py --rows 7 --cols 7 --square-mm 22
"""

import argparse
import os


def generate_svg(rows_inner, cols_inner, square_mm, margin_mm):
    """Genere un SVG vectoriel du damier."""
    rows_squares = rows_inner + 1
    cols_squares = cols_inner + 1
    board_w_mm = cols_squares * square_mm
    board_h_mm = rows_squares * square_mm
    total_w_mm = board_w_mm + 2 * margin_mm
    total_h_mm = board_h_mm + 2 * margin_mm

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w_mm}mm" '
        f'height="{total_h_mm}mm" viewBox="0 0 {total_w_mm} {total_h_mm}">',
        f'<rect x="0" y="0" width="{total_w_mm}" height="{total_h_mm}" fill="white"/>',
    ]

    for r in range(rows_squares):
        for c in range(cols_squares):
            if (r + c) % 2 == 0:
                x = margin_mm + c * square_mm
                y = margin_mm + r * square_mm
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{square_mm}" height="{square_mm}" fill="black"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts), total_w_mm, total_h_mm


def main():
    parser = argparse.ArgumentParser(description="Genere un damier de calibration imprimable (SVG).")
    parser.add_argument("--rows", type=int, default=7, help="Coins internes en hauteur (defaut: 7)")
    parser.add_argument("--cols", type=int, default=7, help="Coins internes en largeur (defaut: 7)")
    parser.add_argument("--square-mm", type=float, default=22.0, help="Taille d'un carre en mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Marge blanche autour")
    parser.add_argument("--output-dir", type=str, default="outputs/chessboards")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_name = f"chessboard_{args.cols}x{args.rows}_{args.square_mm:g}mm"
    svg_text, w_mm, h_mm = generate_svg(args.rows, args.cols, args.square_mm, args.margin_mm)
    svg_path = os.path.join(args.output_dir, f"{base_name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg_text)

    print(f"Damier genere : {svg_path}")
    print(f"  Coins internes : {args.cols}x{args.rows}")
    print(f"  Carres         : {args.cols + 1}x{args.rows + 1} de {args.square_mm} mm")
    print(f"  Damier         : {(args.cols + 1) * args.square_mm:.1f} x {(args.rows + 1) * args.square_mm:.1f} mm")
    print(f"  Feuille        : {w_mm:.1f} x {h_mm:.1f} mm (tient sur A4)")
    print()
    print("Impression : ouvrir dans le navigateur, Cmd+P, echelle 100% (PAS 'adapter a la page').")
    print("Verifier au pied a coulisse la taille des carres apres impression.")


if __name__ == "__main__":
    main()
