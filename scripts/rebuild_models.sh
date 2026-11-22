#!/bin/sh
# Inputs: official checkpoint, Imagenette2-160 archive, pinned MNN source, NEW output directory.
set -eu
[ "$#" = 4 ] || { echo "Usage: rebuild_models.sh CHECKPOINT ARCHIVE MNN_SOURCE NEW_OUTPUT" >&2; exit 2; }
s=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=$4
mkdir "$out"
python3 "$s/export_21m.py" "$1" "$out/21m.onnx"
fp="$out/tinyvit-21m-fp32.mnn"
mix="$out/tinyvit-21m-mlp-w8a8.mnn"
python3 -c 'import _tools,sys;_tools.mnnconvert(sys.argv)' -f ONNX --modelFile "$out/21m.onnx" --MNNModel "$fp" --bizCode ECE496
python3 "$s/prepare_21m_data.py" "$2" "$out/data"
python3 - "$s/model21/calibration.json" "$out" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[2]).resolve()
d=json.loads(pathlib.Path(sys.argv[1]).read_text()); d['path']=str(p/'data/train')
(p/'calibration.json').write_text(json.dumps(d))
PY
python3 -c 'import MNN,_tools,sys;_tools.mnnquant(["mnnquant"]+sys.argv[1:])' "$fp" "$out/calibrated.mnn" "$out/calibration.json"
"${CXX:-c++}" -O2 -std=c++17 -I"$3/schema/current" -I"$3/3rd_party/flatbuffers/include" \
    "$s/../mnn-fpga/mnn/app/create_mixed_quantized_model.cpp" -o "$out/mix-builder"
"$out/mix-builder" "$fp" "$out/calibrated.mnn" "$s/model21/stage1-block1.txt" "$mix"
python3 "$s/evaluate_21m.py" "$out" "$out/data" "$out/host-validation.json"
