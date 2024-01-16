#!/bin/bash
set -e

# dir of this script
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
else
    EMSDK_DIR=$1
fi

WASM_ENV_NAME=nxtgm-emscripten
WASM_ENV_PREFIX=$MAMBA_ROOT_PREFIX/envs/$WASM_ENV_NAME

# check if $WASWM_ENV_PREFIX exists
#if false; then
if [ -d "$WASM_ENV_PREFIX" ]; then
    echo "WASM env $WASM_ENV_PREFIX already exists. Skipping."
else
    micromamba create -n $WASM_ENV_NAME \
        -f $THIS_DIR/environment_wasm_python.yml \
        -c https://repo.mamba.pm/emscripten-forge \
        -c https://repo.mamba.pm/conda-forge \
        --yes \
        --platform=emscripten-wasm32
fi


# clear flgs
export LDFLAGS=""
export CFLAGS=""
export CXXFLAGS=""

source $EMSDK_DIR/emsdk_env.sh

# let cmake know where the env is
export PREFIX=$MAMBA_ROOT_PREFIX/envs/$WASM_ENV_NAME
export CMAKE_PREFIX_PATH=$PREFIX
export CMAKE_SYSTEM_PREFIX_PATH=$PREFIX

mkdir -p $THIS_DIR/build_wasm
pushd $THIS_DIR/build_wasm

# get num cores
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    NUM_CORES=$(nproc --all)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
fi

NUMPY_INCLUDE_DIR=$WASM_ENV_PREFIX/lib/python3.11/site-packages/numpy/core/include

if true; then

    cd $THIS_DIR"/.."
    git clone -b wasm_fixes --single-branch https://github.com/DerThorsten/xtensor-python.git
    cd xtensor-python

    mkdir -p build_wasm
    cd build_wasm
    emcmake cmake \
        -DCMAKE_TOOLCHAIN_FILE=$EMSDK_DIR/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON \
        -DCMAKE_INSTALL_PREFIX=$WASM_ENV_PREFIX \
        -Dxtensor_DIR=$WASM_ENV_PREFIX/share/cmake/xtensor \
        -Dxtl_DIR=$WASM_ENV_PREFIX/share/cmake/xtl \
        -Dpybind11_DIR=$WASM_ENV_PREFIX/share/cmake/pybind11 \
        -DNUMPY_INCLUDE_DIRS=$NUMPY_INCLUDE_DIR \
        ..

    emmake make -j$NUM_CORES install
fi

if true; then
    cd $THIS_DIR
    mkdir -p build_wasm
    cd build_wasm
    emcmake cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON \
        -DCMAKE_INSTALL_PREFIX=$WASM_ENV_PREFIX \
        -DBUILD_TESTS=OFF \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_JAVASCRIPT_BINDINGS=OFF \
        -DBUILD_WITH_COVERAGE=OFF \
        -DBUILD_PLUGINS_DISCRETE_GM_OPTIMIZER=ON \
        -DBUILD_PLUGIN_QPBO_KOLMOGOROV=ON \
        -DBUILD_PLUGIN_HOCR_FIX=ON \
        -DBUILD_PLUGIN_MIN_ST_CUT_KOLMOGOROV=ON \
        -DBUILD_PLUGIN_ILP_COIN_CLP=OFF \
        -DBUILD_DOCS=OFF \
        -DZLIB_INCLUDE_DIR=$WASM_ENV_PREFIX/include \
        -DZLIB_LIBRARY=$WASM_ENV_PREFIX/lib/libz.a \
        -DZLIB_ROOT=$WASM_ENV_PREFIX \
        -DZLIB_USE_STATIC_LIBS=ON \
        -Dxtensor_DIR=$WASM_ENV_PREFIX/share/cmake/xtensor \
        -DHIGHS_LIBRARY=$WASM_ENV_PREFIX/lib/libhighs.a \
        -DHIGHS_INCLUDE_DIR=$WASM_ENV_PREFIX/include/highs \
        -Dxtl_DIR=$WASM_ENV_PREFIX/share/cmake/xtl \
        -Dnlohmann_json_DIR=$WASM_ENV_PREFIX/share/cmake/nlohmann_json \
        -Dtl-expected_DIR=$WASM_ENV_PREFIX/share/cmake/tl-expected \
        -Dxplugin_DIR=$WASM_ENV_PREFIX/lib/cmake/xplugin \
        -Dpybind11_DIR=$WASM_ENV_PREFIX/share/cmake/pybind11 \
        -Dxtensor_DIR=$WASM_ENV_PREFIX/share/cmake/xtensor \
        -Dxtensor-python_DIR=$WASM_ENV_PREFIX/lib/cmake/xtensor-python \
        -DNUMPY_INCLUDE_DIR=$NUMPY_INCLUDE_DIR \
        -DPYTHON_SITE_PACKAGES=$WASM_ENV_PREFIX/lib/python3.11/site-packages \
        ..
    emmake make -j$NUM_CORES install
fi


if true; then


    XEUS_PYTHON_WASM_ENV_NAME="nxtgm-xeus-python"
    XEUS_PYTHON_WASM_ENV_PREFIX=$MAMBA_ROOT_PREFIX/envs/$XEUS_PYTHON_WASM_ENV_NAME

    if [ -d "$XEUS_PYTHON_WASM_ENV_PREFIX" ]; then
        echo "WASM env $XEUS_PYTHON_WASM_ENV_PREFIX already exists. Skipping."
    else
        micromamba create -n $XEUS_PYTHON_WASM_ENV_NAME \
            -f $THIS_DIR/environment_wasm_xpython.yml \
            -c https://repo.mamba.pm/emscripten-forge \
            -c https://repo.mamba.pm/conda-forge \
            --yes \
            --platform=emscripten-wasm32
    fi

    # atm only dirs can be packed, so we move the shared lib in its own dir
    mkdir -p $WASM_ENV_PREFIX/lib/nxtgm/__extra__
    cp $WASM_ENV_PREFIX/lib/libnxtgm_shared.so $WASM_ENV_PREFIX/lib/nxtgm/__extra__/


    MOUNT_PYTHON_PKG="$WASM_ENV_PREFIX/lib/python3.11/site-packages/nxtgm:/lib/python3.11/site-packages/nxtgm"
    MOUNT_LIBNXTGM="$WASM_ENV_PREFIX/lib/nxtgm/__extra__/:/lib/"
    MOUNT_NXTGM_PLUGINS="$WASM_ENV_PREFIX/lib/nxtgm/plugins:/lib/nxtgm/plugins"

    # jupyterlite
    jupyter lite build \
        --contents  $THIS_DIR/python/notebooks/ \
        --XeusAddon.prefix=$XEUS_PYTHON_WASM_ENV_PREFIX \
        --XeusAddon.mounts=$MOUNT_PYTHON_PKG,$MOUNT_LIBNXTGM,$MOUNT_NXTGM_PLUGINS

fi
