# nxtgm

`nxtgm` is C++ 17 library for optimization of discrete graphical models.
It is the inofficial successor of [OpenGM](https://github.com/opengm/opengm)

## Build Status
[![build](https://github.com/nxtgm/nxtgm/actions/workflows/ci_splitted.yaml/badge.svg)](https://github.com/nxtgm/nxtgm/actions/workflows/ci_splitted.yaml)
[![dev-build](https://github.com/nxtgm/nxtgm/actions/workflows/ci.yaml/badge.svg)](https://github.com/nxtgm/nxtgm/actions/workflows/ci.yaml)
[![emscripten](https://github.com/nxtgm/nxtgm/actions/workflows/emscripten.yaml/badge.svg)](https://github.com/nxtgm/nxtgm/actions/workflows/emscripten.yaml)


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
* Hungarian Matching

## Meta-Optimizers
Meta optimizers that use other optimizers

* Partial Optimality Reduced Optimization
* Chained Optimization
* Fusion Moves

## Building blocks for optimizers
* QPBO
* Min-St-Cut / Maxflow
* Higher order clique reduction
* Fusion Moves
* Hungarian Matching / Assignment Problem


## Supported operation systems / architectures
* Linux (x86-64)
* MacOS (x86-64, apple-silicon-arm64)
* Windows (x86_64)
* WebAssembly (emscripten-wasm-32)

## Language bindings
`nxtgm` has language bindings for `python` and `javascript`.
The `python` bindings are generated using `pybind11` and the `javascript` bindings are generated using `emscripten` / `embind`.

### Example

Below is an example of how to use `nxtgm` in C++, Python and Javascript.
The example shows how to create a discrete graphical model with 4 variables and 2 labels per variable. The unary factors are random and the pairwise factors are all the same potts function.
We optimize the graphical model using loopy belief propagation.
This example should illustrate that `nxtgm` has a homogeneous interface for all languages.
Note that this example is not meant to be a good example for discrete graphical models. It is just meant to illustrate the interface.

#### C++
```C++
auto num_var = 4;
auto num_labels = 2;
auto gm = nxtgm::DiscreteGm(4, 2);

// unary factors (with random values, just for the example)
for(auto vi = 0; vi < num_var; ++vi) {
    auto tensor = xt::random::rand<double>({n_labels}, 0.0, 1.0);
    auto func = std::make_unique<nxtgm::XArray>(tensor);
    auto func_index = gm.add_energy_function(std::move(f));
    gm.add_factor({vi}, func_index);
}

// pairwise factors along a chain, all with the same potts function
auto potts_func = std::make_unique<nxtgm::Potts>(num_labels, 0.5);
auto potts_func_index = gm.add_energy_function(std::move(potts_func));
for(auto vi0=0; vi0 < num_var - 1; ++vi0) {
    gm.add_factor({vi0, vi0+1}, potts_func_index);
}

// optimizer
auto optimizer_name = std::string("belief_propagation");
auto parameters = nxtgm::OptimizerParameters();
parameters["max_iterations"] = 10;
parameters["damping"] = 0.9;
parameters["convergence_tolerance"] = 1e-2;
auto optimizer = nxtgm::discrete_gm_optimizer_factory(gm, optimizer_name, parameters);

// optimize
auto status = optimizer.optimize();
auto solution = optimizer.best_solution();
```

#### Python
```python
num_var = 4
num_labels = 2
gm = nxtgm.DiscreteGm(num_var, num_labels)

# unary factors (with random values, just for the example)
for vi in range(num_var):
    unaries = np.random.rand(num_labels)
    func_index = gm.add_energy_function(func)
    gm.add_factor([vi], func_index)

# pairwise factors along a chain, all with the same potts function
potts_func = nxtgm.Potts(num_labels, 0.5)
potts_func_index = gm.add_energy_function(potts_func)
for vi0 in range(num_var - 1):
    gm.add_factor([vi0, vi0+1], potts_func_index)

# optimizer
optimizer_name = "belief_propagation"
parameters = dict(
    max_iterations=10,
    damping=0.9,
    convergence_tolerance=1e-2
)
optimizer = nxtgm.discrete_gm_optimizer_factory(gm, optimizer_name, parameters)

# optimize
status = optimizer.optimize()
solution = optimizer.best_solution()
```

#### Javascript
```javascript
let num_var = 4;
let num_labels = 2;
let gm = new nxtgm.DiscreteGm(num_var, num_labels);

// unary factors (with random values, just for the example)
for(let vi = 0; vi < num_var; ++vi) {
    let unaries = new Float64Array([
        Math.random(), Math.random()
    ]);
    let func = new nxtgm.XArray([num_labels], unaries);
    let func_index = gm.add_energy_function(func);
    gm.add_factor([vi], func_index);
}

// pairwise factors along a chain, all with the same potts function
let potts_func = new nxtgm.Potts(num_labels, 0.5);
let potts_func_index = gm.add_energy_function(potts_func);
for(let v0=0; v0 < num_var - 1; ++v0) {
    gm.add_factor([v0, v0+1], potts_func_index);
}

// optimizer
let optimizer_name = "belief_propagation";
let parameters = new nxtgm.OptimizerParameters();
parameters.set_int("max_iterations", 10);
parameters.set_double("damping", 0.9);
parameters.set_double("convergence_tolerance", 1e-2);
let optimizer = nxtgm.discrete_gm_optimizer_factory(gm, optimizer_name, parameters);

// optimize
let status = optimizer.optimize();
let solution = optimizer.best_solution();
```

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


# Licencing
The `nxtgm` library itself (ie the shared libraries and the header files) are licensed under the MIT license.
See [LICENSE_LIBRARY](LICENSE_LIBRARY) for more details.
The plugins are licensed under different licenses. See [LICENSE_PLUGINS](LICENSE_PLUGINS.md) for more details.


# Why does nxtgm exist?

* Discrete graphical models used to be importat for computer vision. However, nowadays, deep learning is used for many tasks. Nevertheless, discrete graphical models are still important for some niche applications. For example, in the fields where extem little data / ground truth is available, discrete graphical models are still applicable. Or for extremly combinatorial problems,or highly constraint problems , ie matching problems, discrete graphical models are still very useful.

* Since ~2017 [OpenGM](https://github.com/opengm/opengm) is  not maintained anymore and is beyond repairablity (I can say that, I am one of the main authors of OpenGM). The main reason for this is that OpenGM is written in C++03 and uses a lot of template meta programming. This makes is very hard to generate language bindings for other languages and package opengm properly.

* With the rise of tools like copilot, it was easy to port big parts of OpenGM to this  library.

* I wanted to learn more about plugin architectures for C++ and felt discrete graphical models are an excellent use case for this.

* I want to give non-experts the possibility to use discrete graphical models.

* With emscripten it is possible to compile C++ code to WebAssembly. This allows to run discrete graphical models in the browser. This is very useful for teaching purposes.
