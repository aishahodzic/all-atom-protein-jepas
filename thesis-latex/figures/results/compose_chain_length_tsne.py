"""Extract and combine the two original chain-length t-SNE panels losslessly.

No plotting library is needed: this decodes and re-encodes the source RGBA PNGs,
so the scatter points and colour bars are copied pixel-for-pixel.
"""

from pathlib import Path
import argparse
import struct
import zlib


ROOT = Path(__file__).resolve().parents[3]
UNCROPPED = ROOT / "Evaluation/latent_viz/figures/canonical_lejepa/tsne_canonical_lejepa_v1_ema_10k-v-30k_position-and-length.png"
CROPPED = ROOT / "Evaluation/latent_viz/figures/tsne_position_canonical-crop-fixed96-four-ema-10k_grid.png"
OUTPUT = Path(__file__).with_name("chain_length_tsne_uncropped_vs_cropped_ema10k.png")
RAW_UNCROPPED = ROOT / "Evaluation/latent_viz/figures/canonical_lejepa/main_tsne_canonical_lejepa_nocrop_raw_rotation_10k-100k_position-and-length.png"
RAW_CROPPED = ROOT / "Evaluation/latent_viz/figures/canonical_lejepa/main_tsne_canonical_lejepa_crop_protein_rotation_middle_lr_raw_10k_15980_position-and-length.png"
RAW_OUTPUT = Path(__file__).with_name("chain_length_tsne_uncropped_vs_cropped_raw10k.png")
RAW_NO_ROT_CROPPED = ROOT / "Evaluation/latent_viz/figures/canonical_lejepa/tsne_canonical_raw10_middle-lr_protein-residue-rotation-2x2_position.png"
RAW_NO_ROT_OUTPUT = Path(__file__).with_name("chain_length_tsne_uncropped_vs_cropped_raw10k_no_rotation.png")


def read_rgba(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, compressed = 8, bytearray()
    while pos < len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + size]
        pos += size + 12
        if tag == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            assert (depth, color, compression, filtering, interlace) == (8, 6, 0, 0, 0), path
        elif tag == b"IDAT":
            compressed.extend(payload)
        elif tag == b"IEND":
            break
    raw = zlib.decompress(compressed)
    stride = 4 * width
    rows, previous, pos = [], bytearray(stride), 0
    for _ in range(height):
        kind = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += stride + 1
        for i in range(stride):
            left = row[i - 4] if i >= 4 else 0
            above = previous[i]
            upper_left = previous[i - 4] if i >= 4 else 0
            if kind == 1:
                predictor = left
            elif kind == 2:
                predictor = above
            elif kind == 3:
                predictor = (left + above) // 2
            elif kind == 4:
                value = left + above - upper_left
                distances = (abs(value - left), abs(value - above), abs(value - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            elif kind == 0:
                predictor = 0
            else:
                raise ValueError(f"Unsupported PNG filter {kind}")
            row[i] = (row[i] + predictor) & 255
        rows.append(row)
        previous = row
    return width, height, rows


def chunk(tag, payload):
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def write_rgba(path, width, rows):
    height = len(rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    scanlines = b"".join(b"\x00" + bytes(row) for row in rows)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines, 6)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="Combine the raw 10k rotation-matched panels")
    parser.add_argument("--no-rotation", action="store_true", help="Combine the raw 10k no-rotation panels")
    args = parser.parse_args()
    if args.raw and args.no_rotation:
        parser.error("choose only one comparison")
    left_path = RAW_UNCROPPED if args.raw or args.no_rotation else UNCROPPED
    right_path = RAW_NO_ROT_CROPPED if args.no_rotation else (RAW_CROPPED if args.raw else CROPPED)
    output_path = RAW_NO_ROT_OUTPUT if args.no_rotation else (RAW_OUTPUT if args.raw else OUTPUT)
    _, _, left_source = read_rgba(left_path)
    _, _, right_source = read_rgba(right_path)
    # Select only the chain-length panels. For raw comparisons, omit the left
    # colour bar so the shared chain-length scale appears once, on the right.
    x0, x1 = 1940, 2878
    left_y0, left_y1 = (950, 1750) if args.raw else ((195, 915) if args.no_rotation else (115, 915))
    right_y0, right_y1 = (280, 1000) if args.no_rotation else ((115, 915) if args.raw else (950, 1690))
    left_x1 = 2700 if args.raw or args.no_rotation else x1
    left_width = left_x1 - x0
    right_width = x1 - x0
    gap = 36
    height = left_y1 - left_y0
    white = bytes((255, 255, 255, 255))
    rows = []
    for y in range(height):
        left = left_source[left_y0 + y][x0 * 4 : left_x1 * 4]
        right = right_source[right_y0 + y][x0 * 4 : x1 * 4] if y < right_y1 - right_y0 else white * right_width
        rows.append(left + white * gap + right)
    write_rgba(output_path, left_width + right_width + gap, rows)
    print(output_path)


if __name__ == "__main__":
    main()
