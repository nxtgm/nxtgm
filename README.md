# nxtgm

`nxtgm` next graphical models -- a library for optimization of discrete graphical models.
Is the inofficial successor of [OpenGM](https://github.com/opengm/opengm)

## Build Status
[![dev-build](https://github.com/DerThorsten/nxtgm/actions/workflows/ci.yaml/badge.svg)](https://github.com/DerThorsten/nxtgm/actions/workflows/ci.yaml)
[![build](https://github.com/DerThorsten/nxtgm/actions/workflows/ci_splitted.yaml/badge.svg)](https://github.com/DerThorsten/nxtgm/actions/workflows/ci_splitted.yaml)
[![emscripten](https://github.com/DerThorsten/nxtgm/actions/workflows/emscripten.yaml/badge.svg)](https://github.com/DerThorsten/nxtgm/actions/workflows/emscripten.yaml)


# Features

## Optimizers
* Loopy Belief Propagation
* Dynamic Programming
* Brute Force
* Iterated Conditional Modes (ICM)
* Graph-Cut
* QPBO
* Higher order QPBO
* Integer Linear Programming (ILP)

## Meta-Optimizers
Meta optimizers that use other optimizers

* Partial Optimality Reduced Optimization
* Chained Optimization

## Building blocks for optimizers
* QPBO
* Min-St-Cut / Maxflow
* Higher order clique reduction


## Supported operation systems / architectures
* Linux (x86-64)
* MacOS (x86-64, apple-silicon-arm64)
* Windows (x86_64)
* WebAssembly (emscripten-wasm-32)

## Language bindings
`nxtgm` has language bindings for `python` and `javascript`.
The `python` bindings are generated using `pybind11` and the `javascript` bindings are generated using `emscripten` / `embind`.

# Design

## Differences to OpenGM

* runtime polymorphism instead of template meta programming for the value-tables / tensors / functions. This makes it easier to generate language bindings for other languages and simplifies the packaging.

* plugin architecture for optimizers. This makes it easier to add new optimizers and simplifies the packaging. Furthermore optimizers can depend on other optimizers without relying on templates. This allows to generate language bindings for other languages.

* extensive testing. [OpenGM](https://github.com/opengm/opengm) had almost no tests beyond second order graphical models. `nxtgm` has tests for a broad range of graphical models.

## Plugin architecture
nxtgm has a plugin architecture. That means each optimizer is a
plugin. The plugins system we use is [xplugin](https://github.com/quantstack/xplugin) where each plugin is implemented as a shared / dynamic library.

This plugin-based architecture allows to easily add new optimizers to nxtgm
and simplifies the packaging. Instead of depending on concrete implementations
with potential bad licensing, we can just depend on the plugin interface.

Furthermore, optimizers can depend on other plugin interfaces. For example, the `graph-cut` optimizer uses on the  `min_st_cut` plugins

## Plugin interfaces
We define multiple plugin interfaces:

* **`discrete_gm_optimizers`**: an interface for discrete graphical model optimizers (e.g. loopy belief propagation, icm, brute force)

* **`qpbo`**: an interface for `QPBO`-like optimizers. At the moment, there is only one implementation of this interface, which is `QPBO` from Vladimir Kolmogorov.

* **`min_st_cut`**: an interface for `maxflow`/`min-st-cut`-like optimizers. At the moment, there is only one implementation of this interface, which is `maxflow` from Vladimir Kolmogorov.

* **`horc`**: is an interface for higher order clique reduction. At the moment, there is only one implementation of this interface, which is `horc` from  Alexander Fix.

# Why does nxtgm exist?

* Discrete graphical models used to be importat for computer vision. However, nowadays, deep learning is used for many tasks. Nevertheless, discrete graphical models are still important for some niche applications. For example, in the fields where extem little data / ground truth is available, discrete graphical models are still applicable. Or for extremly combinatorial problems,or highly constraint problems , ie matching problems, discrete graphical models are still very useful.

* Since ~2017 [OpenGM](https://github.com/opengm/opengm) is  not maintained anymore and is beyond repairablity (I can say that, I am one of the main authors of OpenGM). The main reason for this is that OpenGM is written in C++03 and uses a lot of template meta programming. This makes is very hard to generate language bindings for other languages and package opengm properly.

* With the rise of tools like copilot, it was easy to port big parts of OpenGM to this  library.

* I wanted to learn more about plugin architectures for C++ and felt discrete graphical models are an excellent use case for this.

* I want to give non-experts in C++, like **biologists** the possibility to use discrete graphical models.

* With emscripten it is possible to compile C++ code to WebAssembly. This allows to run discrete graphical models in the browser. This is very useful for teaching purposes.
