#!/bin/bash
set -e


if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

# dir of this script itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BUILD_DIR=$1
JS_SRC_DIR=$2
JS_DIST_DIR=$3
EMSCRIPTEN_FORGE_EMSDK_DIR=$4

mkdir -p $BUILD_DIR
mkdir -p $JS_DIST_DIR

# abs path to build dir
BUILD_DIR=$(cd $BUILD_DIR && pwd)
NUM_CORES=8

ENV_NAME=nxtgm-emscripten

if true; then

    # install wasm env
    rm -rf $MAMBA_ROOT_PREFIX/envs/$ENV_NAME
    $MAMBA_EXE create -n $ENV_NAME \
        --platform=emscripten-wasm32 \
        -c https://repo.mamba.pm/emscripten-forge \
        -c https://repo.mamba.pm/conda-forge \
        --yes \
        -f $SCRIPT_DIR/environment.yml

fi

if true; then
    export LDFLAGS=""
    export CFLAGS=""
    export CXXFLAGS=""

    #source ~/src/emsdk/emsdk_env.sh
    source $EMSCRIPTEN_FORGE_EMSDK_DIR/emsdk_env.sh


    # let cmake know where the env is
    export PREFIX=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME
    export CMAKE_PREFIX_PATH=$PREFIX
    export CMAKE_SYSTEM_PREFIX_PATH=$PREFIX

    pushd $BUILD_DIR
    # build pyjs
    emcmake cmake \
        -DCMAKE_TOOLCHAIN_FILE=$EMSCRIPTEN_FORGE_EMSDK_DIR/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DBUILD_TESTS=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF \
        -DBUILD_JAVASCRIPT_BINDINGS=ON \
        -DBUILD_WITH_COVERAGE=OFF \
        -DBUILD_PLUGINS_DISCRETE_GM_OPTIMIZER=ON \
        -DBUILD_PLUGIN_QPBO_KOLMOGOROV=ON \
        -DBUILD_PLUGIN_HOCR_FIX=ON \
        -DBUILD_PLUGIN_MIN_ST_CUT_KOLMOGOROV=ON \
        -DBUILD_PLUGIN_ILP_COIN_CLP=OFF \
        -DBUILD_DOCS=OFF \
        -DZLIB_INCLUDE_DIR=$PREFIX/include \
        -DZLIB_LIBRARY=$PREFIX/lib/libz.a \
        -DZLIB_ROOT=$PREFIX \
        -DZLIB_USE_STATIC_LIBS=ON \
        -Dxtensor_DIR=$PREFIX/share/cmake/xtensor \
        ..
    popd

    pushd $BUILD_DIR

    emmake make -j$NUM_CORES install
    popd

fi




cp -r $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/lib/nxtgm/plugins   $JS_DIST_DIR/plugins
cp $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/lib/libnxtgm_shared.so $JS_DIST_DIR/libnxtgm_shared.so
cp $BUILD_DIR/javascript/nxtgm_javascript_runtime.*         $JS_DIST_DIR
cp $BUILD_DIR/javascript/nxtgm_javascript_runtime.*         $JS_SRC_DIR
