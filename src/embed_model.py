#!/usr/bin/env python3
"""Embed the locally constructed and schema-verified MLP model using only the Python stdlib."""
import hashlib
from pathlib import Path
import sys


def generate(model, output):
    raw = model.read_bytes()
    if raw[4:8] != b"TFL3":
        raise ValueError("Not a TFLite model")
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_data.h").write_text(
        "#pragma once\nextern const unsigned char g_model[];\n"
        "extern const unsigned int g_model_len;\n", encoding="utf-8")
    rows = ["    " + ", ".join("0x%02x" % b for b in raw[i:i+16]) + ","
            for i in range(0, len(raw), 16)]
    (output / "model_data.cc").write_text(
        '#include "model_data.h"\nalignas(16) const unsigned char g_model[] = {\n'
        + "\n".join(rows) + "\n};\nconst unsigned int g_model_len = "
        + str(len(raw)) + ";\n", encoding="utf-8")
    print("Embedded model: %d bytes" % len(raw))


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
