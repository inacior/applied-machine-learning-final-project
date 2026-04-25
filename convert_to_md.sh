#!/bin/bash
# Convert all PDFs in books/ and papers/ to Markdown using Docling (IBM)
# Usage: ./convert_to_md.sh

set -e

export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SCRIPT_DIR/papers/markdown_extraction"
mkdir -p "$SCRIPT_DIR/books/markdown_extraction"

echo "=== Converting Papers ==="
for pdf in "$SCRIPT_DIR"/papers/*.pdf; do
    basename=$(basename "$pdf" .pdf)
    echo "Processing: $basename"
    docling "$pdf" --to md --output "$SCRIPT_DIR/papers/markdown_extraction" 2>&1
    echo "Done: $basename"
    echo ""
done

echo "=== Converting Books ==="
for pdf in "$SCRIPT_DIR"/books/*.pdf; do
    basename=$(basename "$pdf" .pdf)
    echo "Processing: $basename"
    docling "$pdf" --to md --output "$SCRIPT_DIR/books/markdown_extraction" 2>&1
    echo "Done: $basename"
    echo ""
done

echo "=== All done! ==="
