

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

// xtensor view
#include <xtensor/xview.hpp>

namespace nxtgm{


    template<class XTENSOR>
    auto bind_at(const XTENSOR & xtensor, std::size_t axis, std::size_t value){
        xt::xstrided_slice_vector sv(xtensor.dimension(), xt::all());
        sv[axis] = value;
        return  xt::strided_view(xtensor, sv);
    }

    
    void IlpFactorBuilderBuffer::ensure_size(std::size_t max_factor_size, std::size_t max_factor_arity)
    {
        if(energy_buffer.size() < max_factor_size){
            energy_buffer.resize(max_factor_size*2);
        }
        if(label_buffer.size() < max_factor_arity){
            label_buffer.resize(max_factor_arity*2);
        }
        if(shape_buffer.size() < max_factor_arity){
            shape_buffer.resize(max_factor_arity*2);
        }
    }

    energy_type DiscreteEnergyFunctionBase::energy(std::initializer_list<discrete_label_type> discrete_labels) const {
        return this->energy( discrete_labels.begin());
    }
 
    std::size_t DiscreteEnergyFunctionBase::size() const {
        std::size_t result = 1;
        for(std::size_t i = 0; i < arity(); ++i){
            result *= static_cast<std::size_t>(shape(i));
        }
        return result;
    }

    void  DiscreteEnergyFunctionBase::copy_energies(
        energy_type * energies,
        discrete_label_type * discrete_labels_buffer
    ) const {
        const auto arity = this->arity();
        auto shapef = [this](std::size_t index) { return this->shape(index); };

        auto flat_index = 0;
        n_nested_loops<discrete_label_type>(arity, shapef, discrete_labels_buffer, [&](auto && _){
            energies[flat_index] = this->energy(discrete_labels_buffer);
            ++flat_index;
        });
    }

    void  DiscreteEnergyFunctionBase::add_energies(
        energy_type * energies,
        discrete_label_type * discrete_labels_buffer
    ) const {
        const auto arity = this->arity();
        auto shapef = [this](std::size_t index) { return this->shape(index); };

        auto flat_index = 0;
        n_nested_loops<discrete_label_type>(arity, shapef, discrete_labels_buffer, [&](auto && _){
            
            energies[flat_index] += this->energy(discrete_labels_buffer);
            ++flat_index;
        });
    }

    void DiscreteEnergyFunctionBase::copy_shape(discrete_label_type * shape) const {
        for(std::size_t i = 0; i < arity(); ++i){
            shape[i] = this->shape(i);
        }
    }

    void DiscreteEnergyFunctionBase::add_to_lp(
        IlpData & ilp_data, 
        const std::size_t * indicator_variables_mapping,
        IlpFactorBuilderBuffer & buffer
    ) const
    {
        const auto arity = this->arity();
        const auto factor_size = this->size();

        // make sure the buffer is big enough
        buffer.ensure_size(factor_size, arity);

        auto labels = buffer.label_buffer.data();
        auto energies =buffer.energy_buffer.data();

        this->copy_energies(buffer.energy_buffer.data(), buffer.label_buffer.data());

        if(arity == 1)
        {
            for(discrete_label_type label = 0; label < static_cast<discrete_label_type>(factor_size); ++label)
            {
                ilp_data[indicator_variables_mapping[0] + label] += energies[label];
            }
        }
        else
        {
            auto shape = buffer.shape_buffer.data();
            this->copy_shape(shape);

            // where to the factor indicator variables start?
            const auto start = ilp_data.num_variables();
            // todo: avoid allocation? 
            auto factor_indicator_vars = xt::eval(xt::arange(start, start + factor_size).reshape(
                const_discrete_label_span(shape, arity)
            ));

            ilp_data.add_variables(0.0, 1.0, energies, energies + factor_size,false);


            for(auto ai=0; ai<arity; ++ai)
            {

                for(auto label=0; label<shape[ai]; ++label)
                {
                    const auto var_inidcator = indicator_variables_mapping[ai] + label;
                    auto constraint_vars = bind_at(factor_indicator_vars, ai, label);
                    
                    ilp_data.begin_constraint(0.0, 0.0);
                    ilp_data.add_constraint_coefficient(var_inidcator, -1.0);
                    for(auto var : constraint_vars){
                        ilp_data.add_constraint_coefficient(var, 1.0);
                    }
                }
            }
        }
    }
}