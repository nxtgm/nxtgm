#pragma once

#ifndef EMSCRIPTEN
#include <iostream>
#else
#include <emscripten.h>
#endif

#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <test.hpp>

namespace nxtgm::tests
{

template <class F>
void for_each(std::vector<discrete_label_type> &shape, F &&f)
{
    std::vector<discrete_label_type> labels(shape.size(), 0);

    using L = discrete_label_type;

    switch (shape.size())
    {
    case 1:
        for (L l0 = 0; l0 < shape[0]; ++l0)
        {
            labels[0] = l0;
            f(labels.data());
        }
        break;
    case 2:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
            {
                labels[0] = l0;
                labels[1] = l1;
                f(labels.data());
            }
        break;
    case 3:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                {
                    labels[0] = l0;
                    labels[1] = l1;
                    labels[2] = l2;
                    f(labels.data());
                }
        break;
    case 4:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                    {
                        labels[0] = l0;
                        labels[1] = l1;
                        labels[2] = l2;
                        labels[3] = l3;
                        f(labels.data());
                    }
        break;
    case 5:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                        {
                            labels[0] = l0;
                            labels[1] = l1;
                            labels[2] = l2;
                            labels[3] = l3;
                            labels[4] = l4;
                            f(labels.data());
                        }
        break;
    case 6:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                            for (L l5 = 0; l5 < shape[5]; ++l5)
                            {
                                labels[0] = l0;
                                labels[1] = l1;
                                labels[2] = l2;
                                labels[3] = l3;
                                labels[4] = l4;
                                labels[5] = l5;
                                f(labels.data());
                            }
        break;
    case 7:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                            for (L l5 = 0; l5 < shape[5]; ++l5)
                                for (L l6 = 0; l6 < shape[6]; ++l6)
                                {
                                    labels[0] = l0;
                                    labels[1] = l1;
                                    labels[2] = l2;
                                    labels[3] = l3;
                                    labels[4] = l4;
                                    labels[5] = l5;
                                    labels[6] = l6;
                                    f(labels.data());
                                }
        break;
    case 8:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                            for (L l5 = 0; l5 < shape[5]; ++l5)
                                for (L l6 = 0; l6 < shape[6]; ++l6)
                                    for (L l7 = 0; l7 < shape[7]; ++l7)
                                    {
                                        labels[0] = l0;
                                        labels[1] = l1;
                                        labels[2] = l2;
                                        labels[3] = l3;
                                        labels[4] = l4;
                                        labels[5] = l5;
                                        labels[6] = l6;
                                        labels[7] = l7;
                                        f(labels.data());
                                    }
        break;
    case 9:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                            for (L l5 = 0; l5 < shape[5]; ++l5)
                                for (L l6 = 0; l6 < shape[6]; ++l6)
                                    for (L l7 = 0; l7 < shape[7]; ++l7)
                                        for (L l8 = 0; l8 < shape[8]; ++l8)
                                        {
                                            labels[0] = l0;
                                            labels[1] = l1;
                                            labels[2] = l2;
                                            labels[3] = l3;
                                            labels[4] = l4;
                                            labels[5] = l5;
                                            labels[6] = l6;
                                            labels[7] = l7;
                                            labels[8] = l8;
                                            f(labels.data());
                                        }
        break;
    case 10:
        for (L l0 = 0; l0 < shape[0]; ++l0)
            for (L l1 = 0; l1 < shape[1]; ++l1)
                for (L l2 = 0; l2 < shape[2]; ++l2)
                    for (L l3 = 0; l3 < shape[3]; ++l3)
                        for (L l4 = 0; l4 < shape[4]; ++l4)
                            for (L l5 = 0; l5 < shape[5]; ++l5)
                                for (L l6 = 0; l6 < shape[6]; ++l6)
                                    for (L l7 = 0; l7 < shape[7]; ++l7)
                                        for (L l8 = 0; l8 < shape[8]; ++l8)
                                            for (L l9 = 0; l9 < shape[9]; ++l9)
                                            {
                                                labels[0] = l0;
                                                labels[1] = l1;
                                                labels[2] = l2;
                                                labels[3] = l3;
                                                labels[4] = l4;
                                                labels[5] = l5;
                                                labels[6] = l6;
                                                labels[7] = l7;
                                                labels[8] = l8;
                                                labels[9] = l9;
                                                f(labels.data());
                                            }
        break;

    default:
        throw std::runtime_error("not implemented for dimension > 10");
    }
}

template <class T>
inline void test_discrete_constraint_function(DiscreteConstraintFunctionBase *f)
{
    const auto arity = f->arity();
    CHECK(arity > 0);

    // get the shape and compute the size
    std::size_t size = 1;
    std::vector<discrete_label_type> shape(f->arity());
    for (std::size_t i = 0; i < f->arity(); ++i)
    {
        shape[i] = f->shape(i);
        size *= shape[i];
    }

    // size sanity check
    CHECK(f->size() == size);

    // serialize to json
    auto as_json = f->serialize_json();

    // check that "type" is in the json and that json is an object
    CHECK(as_json.is_object());
    CHECK(as_json.contains("type"));

    // check that type is a string
    CHECK(as_json["type"].is_string());

    std::string type = as_json["type"];

    CHECK(type == T::serialization_key());

    // deserialize via factory function
    {
        auto f_j = nxtgm::discrete_constraint_function_deserialize_json(as_json);

        auto i = 0;
        for_each(shape, [&](auto labels) {
            const auto is_value = f->how_violated(labels);
            const auto should_value = f_j->how_violated(labels);
            if (!CHECK(is_value == doctest::Approx(should_value)))
            {

                std::stringstream ss;
                ss << "ERROR: how_violated() is consistent with json "
                      "serialized+deserialized"
                   << std::endl;
                for (auto i = 0; i < arity; ++i)
                {
                    ss << labels[i] << " ";
                }
                ss << " -> IS " << is_value << " != SHOULD BE " << should_value << std::endl;

#ifdef EMSCRIPTEN
                emscripten_log(EM_LOG_ERROR, ss.str().c_str()));
#else
                std::cout << ss.str() << std::endl;
#endif
            }
            ++i;
        });
    }
}

template <class T>
inline void test_discrete_energy_function(DiscreteEnergyFunctionBase *f)
{
    const auto arity = f->arity();
    CHECK(arity > 0);

    // get the shape and compute the size
    std::size_t size = 1;
    std::vector<discrete_label_type> shape(f->arity());
    for (std::size_t i = 0; i < f->arity(); ++i)
    {
        shape[i] = f->shape(i);
        size *= shape[i];
    }

    // size sanity check
    CHECK(f->size() == size);

    // copy energies
    std::vector<energy_type> energies_copy(size, 0);
    std::vector<energy_type> energies_copy_should(size, 0);
    f->copy_energies(energies_copy.data());

    // add energies
    std::vector<energy_type> energies_sum(size, 1.0);
    f->add_energies(energies_sum.data());

    // check that copy_energies and add_energies are consistent
    for (std::size_t i = 0; i < size; ++i)
    {
        CHECK(energies_copy[i] + 1 == doctest::Approx(energies_sum[i]));
    }

    auto i = 0;
    for_each(shape, [&](auto labels) {
        const auto is_value = f->energy(labels);
        if (!CHECK(is_value == doctest::Approx(energies_copy[i])))
        {
            std::cout << "ERROR: energies() ... consistent with copy_energies" << std::endl;
            for (auto i = 0; i < arity; ++i)
            {
                std::cout << labels[i] << " ";
            }
            std::cout << " -> IS " << is_value << " != SHOULD BE " << energies_copy[i] << std::endl;
        }
        ++i;
    });

    // serialize to json
    auto as_json = f->serialize_json();

    // check that "type" is in the json and that json is an object
    CHECK(as_json.is_object());
    CHECK(as_json.contains("type"));

    // check that type is a string
    CHECK(as_json["type"].is_string());

    std::string type = as_json["type"];

    CHECK(type == T::serialization_key());

    // deserialize via static function
    {
        auto f_j = T::deserialize_json(as_json);

        // check that the deserialized version is the same as the original
        CHECK(f_j->size() == f->size());
        CHECK(f_j->arity() == f->arity());
        for (std::size_t i = 0; i < f->arity(); ++i)
        {
            CHECK(f_j->shape(i) == f->shape(i));
        }

        std::vector<energy_type> energies_copy_from_json(size, 0);
        std::vector<energy_type> energies_copy_should_from_json(size, 0);

        f_j->copy_energies(energies_copy_from_json.data());

        // check that copy_energies and energies_copy_from_json are consistent
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK(energies_copy[i] == doctest::Approx(energies_copy_from_json[i]));
        }
    }
    // deserialize via static function
    {
        auto f_j = T::deserialize_json(as_json);

        // check that the deserialized version is the same as the original
        CHECK(f_j->size() == f->size());
        CHECK(f_j->arity() == f->arity());
        for (std::size_t i = 0; i < f->arity(); ++i)
        {
            CHECK(f_j->shape(i) == f->shape(i));
        }

        std::vector<energy_type> energies_copy_from_json(size, 0);
        std::vector<energy_type> energies_copy_should_from_json(size, 0);

        f_j->copy_energies(energies_copy_from_json.data());

        // check that copy_energies and energies_copy_from_json are consistent
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK(energies_copy[i] == doctest::Approx(energies_copy_from_json[i]));
        }
    }
    // deserialize via factory function
    {
        auto f_j = nxtgm::discrete_energy_function_deserialize_json(as_json);

        // check that the deserialized version is the same as the original
        CHECK(f_j->size() == f->size());
        CHECK(f_j->arity() == f->arity());
        for (std::size_t i = 0; i < f->arity(); ++i)
        {
            CHECK(f_j->shape(i) == f->shape(i));
        }

        std::vector<energy_type> energies_copy_from_json(size, 0);
        std::vector<energy_type> energies_copy_should_from_json(size, 0);

        f_j->copy_energies(energies_copy_from_json.data());

        // check that copy_energies and energies_copy_from_json are consistent
        for (std::size_t i = 0; i < size; ++i)
        {
            CHECK(energies_copy[i] == doctest::Approx(energies_copy_from_json[i]));
        }
    }
}

} // namespace nxtgm::tests
