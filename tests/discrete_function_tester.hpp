#pragma once

#include <test.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>

namespace nxtgm::tests{

    void test_energy_function(DiscreteEnergyFunctionBase * f)
    {
        const auto arity = f->arity();
        CHECK(arity > 0);

        // get the shape and compute the size
        std::size_t size = 1;
        std::vector<discrete_label_type> shape(f->arity());
        for(std::size_t i = 0; i < f->arity(); ++i){
            shape[i] = f->shape(i);
            size *= shape[i];
        }

        // size sanity check
        CHECK(f->size() == size);

        std::vector<discrete_label_type> discrete_labels_buffer(arity);

        // copy energies
        std::vector<energy_type> energies_copy(size, 0);
        std::vector<energy_type> energies_copy_should(size, 0);
        f->copy_energies(energies_copy.data(), discrete_labels_buffer.data());

        if(arity == 1){
            for(discrete_label_type i = 0; i < size; ++i){
                energies_copy_should[i] = f->energy({i});
            }
        }


        std::vector<energy_type> energies_sum(size, 0);
        f->add_energies(energies_sum.data(), discrete_labels_buffer.data());

        // check that copy_energies and add_energies are consistent
        for(std::size_t i = 0; i < size; ++i){
            CHECK_EQ(energies_copy[i], energies_sum[i]);
        }

    

        
    }


}