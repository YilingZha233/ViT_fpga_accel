#!/bin/sh
# Rebuild the static ARM stream runner using the team's pinned MNN + Zig toolchain.
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo="$project/mnn-fpga"
base="$project/build"
checkout="$base/MNN"
pin=$(cat "$repo/mnn/PINNED_REVISION")
mkdir -p "$base"
if [ ! -d "$checkout" ]; then
    git init -q "$checkout"
    git -C "$checkout" remote add origin https://github.com/alibaba/MNN.git
fi
if ! git -C "$checkout" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$checkout" fetch --depth=1 origin "$pin"
    git -C "$checkout" checkout --detach FETCH_HEAD
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$pin"
for patch in "$repo"/mnn/patches/*.patch; do
    if ! git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$checkout" apply "$patch"
    fi
done
: "${ZIG_EXE:?Set ZIG_EXE to the absolute path of Zig 0.13.0}"
test "$("$ZIG_EXE" version)" = 0.13.0
export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$base/zig-cache}"
tc="$repo/platform/arm-zig-windows.cmake"
cmake -S "$checkout" -B "$base/lib" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$tc" -DZIG_EXE="$ZIG_EXE" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DMNN_FPGA_ROOT="$repo" -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_HARD=ON \
    -DMNN_BUILD_TOOLS=OFF -DMNN_SEP_BUILD=OFF -DMNN_ARM82=OFF \
    -DMNN_USE_THREAD_POOL=ON -DCMAKE_CXX_STANDARD=17
cmake --build "$base/lib" --target MNN -j "${JOBS:-4}"
cmake -S "$repo/mnn/app" -B "$base/app" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$tc" -DZIG_EXE="$ZIG_EXE" -DMNN_SOURCE="$checkout" \
    -DMNN_BUILD="$base/lib" -DTINYVIT_STATIC=ON
cmake --build "$base/app" --target tinyvit_run -j "${JOBS:-4}"
cp "$base/app/tinyvit_run" "$project/demo/bin/tinyvit_stream.new"
chmod 755 "$project/demo/bin/tinyvit_stream.new"
mv "$project/demo/bin/tinyvit_stream.new" "$project/demo/bin/tinyvit_stream"

"$ZIG_EXE" c++ -target arm-linux-musleabihf -mcpu=cortex_a9 -std=c++17 -O2 -static -s \
    -I"$repo/sw/include" "$repo/sw/src/matmul_soc_native_smoke.cpp" "$repo/sw/src/matmul_soc.cpp" \
    "$repo/sw/src/matmul_soc_native.cpp" "$repo/sw/src/matmul_soc_uio.cpp" -o "$project/demo/bin/ni82_smoke21"
