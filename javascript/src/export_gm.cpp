#include <string>

#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/functions/xarray_energy_function.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include "convert.hpp"

namespace nxtgm
{

namespace em = emscripten;

void export_gm()
{
    em::class_<SolutionValue>("SolutionValue")
        .constructor<energy_type, energy_type>()
        .function("energy",
                  em::select_overload<energy_type(SolutionValue &)>([](SolutionValue &self) { return self.energy(); }))
        .function("how_violated", em::select_overload<energy_type(SolutionValue &)>(
                                      [](SolutionValue &self) { return self.how_violated(); }))
        // .function("how_violated", &SolutionValue::how_violated)
        ;
    em::class_<DiscreteGm>("DiscreteGm")

        .constructor<const DiscreteSpace &>()
        .constructor<std::size_t, std::size_t>()
        .function("save_binary", &DiscreteGm::save_binary)
        .class_function("load_binary", &DiscreteGm::load_binary)
        .function("as_json_str",
                  em::select_overload<std::string(DiscreteGm &, std::size_t)>([](DiscreteGm &self, std::size_t indent) {
                      const auto str = self.serialize_json().dump(indent);
                      return str;
                  }))

        .function("add_energy_function", em::select_overload<std::size_t(DiscreteGm &, DiscreteEnergyFunctionBase &)>(
                                             [](DiscreteGm &self, DiscreteEnergyFunctionBase &f) {
                                                 return self.add_energy_function(std::move(f.clone()));
                                             }))
        .function("add_constraint_function",
                  em::select_overload<std::size_t(DiscreteGm &, DiscreteConstraintFunctionBase &)>(
                      [](DiscreteGm &self, DiscreteConstraintFunctionBase &f) {
                          return self.add_constraint_function(std::move(f.clone()));
                      }))

        // batch api to add functions
        .function("add_xarray_energy_functions",
                  em::select_overload<em::val(DiscreteGm &, em::val)>([](DiscreteGm &gm, em::val ndarray) {
                      std::vector<uint64_t> fids;
                      auto batch_array = copy_from_ndarray<double>(ndarray);

                      const std::size_t n_functions = batch_array.shape()[0];

                      for (std::size_t i = 0; i < n_functions; i++)
                      {
                          auto arr = xt::view(batch_array, i);

                          auto arr_function = std::make_unique<XArray>(std::move(arr));
                          fids.push_back(gm.add_energy_function(std::move(arr_function)));
                      }
                      return ptr_range_to_typed_array_copy(fids.data(), fids.size());
                  }))

        // // batch api to add factors
        // .function("add_factors", em::select_overload< em::val(DiscreteGm &, std::size_t,em::val, em::val)>(

        .function("add_factor", em::select_overload<std::size_t(DiscreteGm &, em::val, std::size_t)>(
                                    [](DiscreteGm &self, em::val variables, std::size_t fid) {
                                        // check that value is list
                                        if (!variables.isArray())
                                        {
                                            throw std::runtime_error("variables is not an array");
                                        }
                                        const auto length = variables["length"].as<int>();
                                        std::vector<std::size_t> vars(length);
                                        // check that value is list of numbers
                                        for (int i = 0; i < length; i++)
                                        {
                                            if (!variables[i].isNumber())
                                            {
                                                throw std::runtime_error("variables is not an array of numbers");
                                            }
                                            vars[i] = variables[i].as<std::size_t>();
                                        }

                                        return self.add_factor(vars, fid);
                                    }))

        .function("add_constraint", em::select_overload<std::size_t(DiscreteGm &, em::val, std::size_t)>(
                                        [](DiscreteGm &self, em::val variables, std::size_t fid) {
                                            // check that value is list
                                            if (!variables.isArray())
                                            {
                                                throw std::runtime_error("variables is not an array");
                                            }
                                            const auto length = variables["length"].as<int>();
                                            std::vector<std::size_t> vars(length);
                                            // check that value is list of numbers
                                            for (int i = 0; i < length; i++)
                                            {
                                                if (!variables[i].isNumber())
                                                {
                                                    throw std::runtime_error("variables is not an array of numbers");
                                                }
                                                vars[i] = variables[i].as<std::size_t>();
                                            }

                                            return self.add_constraint(vars, fid);
                                        }))

        // batch api to add functions
        .function(
            "add_factors",
            em::select_overload<em::val(DiscreteGm &, em::val, em::val)>([](DiscreteGm &gm, em::val vis, em::val fids) {
                // this is a copy, todo: avoid copy
                auto fids_vector = em::convertJSArrayToNumberVector<uint64_t>(fids);
                auto vis_array = copy_from_ndarray<uint32_t>(vis);
                const std::size_t n_factors = vis_array.shape()[0];
                auto factor_indices = std::vector<uint64_t>(n_factors);

                if (n_factors != fids_vector.size() && fids_vector.size() != 1)
                {
                    throw std::runtime_error("vis and fids must have the same length, or fids must have length 1");
                }

                for (std::size_t i = 0; i < n_factors; i++)
                {
                    const auto fid_index = i >= fids_vector.size() ? 0 : i;

                    auto vis = xt::view(vis_array, i);
                    auto fi = gm.add_factor(vis, fids_vector[fid_index]);
                    factor_indices[i] = fi;
                }
                return ptr_range_to_typed_array_copy(factor_indices.data(), factor_indices.size());
            }))

        // readonly properties
        .property("num_variables", &DiscreteGm::num_variables)
        .property("num_factors", &DiscreteGm::num_factors)
        .property("num_constraints", &DiscreteGm::num_constraints)

        .function("max_num_labels",
                  em::select_overload<discrete_label_type(DiscreteGm &)>(
                      [](DiscreteGm &self) -> discrete_label_type { return self.space().max_num_labels(); }))

        .property("max_arity", &DiscreteGm::max_arity)
        .property("max_constraint_arity", &DiscreteGm::max_factor_arity)
        .property("max_constraint_arity", &DiscreteGm::max_constraint_arity)

        .property("max_constraint_size", &DiscreteGm::max_constraint_size)
        .property("max_factor_size", &DiscreteGm::max_factor_size);
}

} // namespace nxtgm
