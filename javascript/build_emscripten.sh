#!/bin/bash
set -e


BUILD_DIR=build_wasm
mkdir -p $BUILD_DIR

SERVER_DIR=serve_dir

# abs path to build dir
BUILD_DIR=$(cd $BUILD_DIR && pwd)
NUM_CORES=8

ENV_NAME=nxtgm-emscripten


EMSCRIPTEN_FORGE_EMSDK_DIR=/Users/thorstenbeier/src/emsdk_custom2

if false; then

    # install wasm env
    rm -rf $MAMBA_ROOT_PREFIX/envs/$ENV_NAME
    $MAMBA_EXE create -n $ENV_NAME \
        --platform=emscripten-wasm32 \
        -c https://repo.mamba.pm/emscripten-forge \
        -c https://repo.mamba.pm/conda-forge \
        -c ~/micromamba/envs/emf/conda-bld \
        --yes \
        xtensor xplugin  nlohmann_json xtl tsl_ordered_map

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
        -DBUILD_DOCS=OFF \
        ..
    popd

    pushd $BUILD_DIR

    emmake make -j$NUM_CORES install
    popd

fi



# copy  files to server dir
cp -r $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/lib/nxtgm/plugins $SERVER_DIR/plugins

# copy shared libs to server dir
cp $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/lib/libnxtgm_shared.so $SERVER_DIR/libnxtgm_shared.so

# copy wasm files to server dir
cp $MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/nxtgm_javascript_runtime.* $SERVER_DIR
