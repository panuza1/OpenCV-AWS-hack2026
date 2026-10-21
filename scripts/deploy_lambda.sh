#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-package}"
ZIP_PATH="${2:-$ROOT_DIR/dist/physicseye-lambda.zip}"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

mkdir -p "$BUILD_DIR/physicseye" "$(dirname "$ZIP_PATH")"
cp "$ROOT_DIR/cloud/lambda_handler.py" "$BUILD_DIR/lambda_handler.py"
cp "$ROOT_DIR/src/physicseye/__init__.py" "$BUILD_DIR/physicseye/__init__.py"
cp "$ROOT_DIR/src/physicseye/physics.py" "$BUILD_DIR/physicseye/physics.py"

"${PYTHON:-python3}" - "$BUILD_DIR" "$ZIP_PATH" <<'PY'
import pathlib
import sys
import zipfile

source, output = map(pathlib.Path, sys.argv[1:])
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*.py")):
        archive.write(path, path.relative_to(source))
print(output)
PY

if [[ "$ACTION" == "package" ]]; then
    exit 0
fi
if [[ "$ACTION" != "deploy" ]]; then
    echo "usage: $0 [package|deploy] [zip-path]" >&2
    exit 2
fi

: "${AWS_REGION:?Set AWS_REGION}"
: "${PHYSICSEYE_LAMBDA_NAME:?Set PHYSICSEYE_LAMBDA_NAME}"
: "${PHYSICSEYE_LAMBDA_ROLE_ARN:?Set PHYSICSEYE_LAMBDA_ROLE_ARN}"

if aws lambda get-function --region "$AWS_REGION" --function-name "$PHYSICSEYE_LAMBDA_NAME" >/dev/null 2>&1; then
    aws lambda update-function-code --region "$AWS_REGION" \
        --function-name "$PHYSICSEYE_LAMBDA_NAME" --zip-file "fileb://$ZIP_PATH"
else
    aws lambda create-function --region "$AWS_REGION" \
        --function-name "$PHYSICSEYE_LAMBDA_NAME" \
        --runtime python3.13 --handler lambda_handler.handler \
        --role "$PHYSICSEYE_LAMBDA_ROLE_ARN" \
        --zip-file "fileb://$ZIP_PATH" --timeout 10 --memory-size 128 \
        --architectures x86_64
fi
