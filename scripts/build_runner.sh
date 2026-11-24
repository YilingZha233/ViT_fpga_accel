#!/bin/sh
# Build the static co-design stream runner with the team's pinned MNN.
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo="$project/mnn-fpga"
base="${CODESIGN_BUILD_DIR:-$project/build/codesign}"
checkout="$base/MNN"
pin=$(tr -d '\015' < "$repo/mnn/PINNED_REVISION")
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
    normalized="$base/$(basename "$patch")"
    tr -d '\015' < "$patch" > "$normalized"
    if ! git -C "$checkout" apply --reverse --check "$normalized" 2>/dev/null; then
        git -C "$checkout" apply "$normalized"
    fi
done
if [ -n "${ZIG_EXE:-}" ]; then
    test "$("$ZIG_EXE" version)" = 0.13.0
    export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-$base/zig-cache}"
    tc="$repo/platform/arm-zig-windows.cmake"
    set -- "-DZIG_EXE=$ZIG_EXE"
else
    tc="$repo/platform/arm-linux-gnueabihf.cmake"
    set --
    # GNU as requires whitespace before the register list.
    sed -i 's/vld1\.32{/vld1.32 {/g' "$checkout/source/backend/cpu/arm/arm32/MNNCubicLineC16.S"
fi
cmake -S "$checkout" -B "$base/lib" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$tc" "$@" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DMNN_FPGA_ROOT="$repo" -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_HARD=ON \
    -DMNN_BUILD_TOOLS=OFF -DMNN_SEP_BUILD=OFF -DMNN_ARM82=OFF \
    -DMNN_USE_THREAD_POOL=ON -DCMAKE_CXX_STANDARD=17
cmake --build "$base/lib" --target MNN -j "${JOBS:-4}"
cmake -S "$repo/mnn/app" -B "$base/app" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$tc" "$@" -DMNN_SOURCE="$checkout" \
    -DMNN_BUILD="$base/lib" -DTINYVIT_STATIC=ON -DCMAKE_EXE_LINKER_FLAGS="-static -s"
cmake --build "$base/app" --target tinyvit_run -j "${JOBS:-4}"
cp "$base/app/tinyvit_run" "$project/demo/bin/tinyvit_stream_codesign.new"
chmod 755 "$project/demo/bin/tinyvit_stream_codesign.new"
mv "$project/demo/bin/tinyvit_stream_codesign.new" "$project/demo/bin/tinyvit_stream_codesign"
