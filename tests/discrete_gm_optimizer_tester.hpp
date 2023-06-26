#pragma once

#include <test.hpp>
#include <testmodels.hpp>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/callbacks.hpp>
#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>

#include <progressbar.hpp>

namespace nxtgm::tests{


inline std::pair<typename DiscreteGm::solution_type, SolutionValue> solve_brute_force(const DiscreteGm& model){

    using gm_type = std::decay_t<decltype(model)>;
    using solution_type = typename gm_type::solution_type;
    using optimizer_type = nxtgm::BruteForceNaive;

    auto optimizer_parameter = typename optimizer_type::parameter_type();
    auto optimizer = std::make_unique<optimizer_type>(model, optimizer_parameter);

    optimizer->optimize( );
    return std::pair<solution_type, SolutionValue>(optimizer->best_solution(), optimizer->best_solution_value());
}



struct CheckFeasiblity
{
    static std::string name(){ return "CheckFeasiblity"; }
    void check(const DiscreteGm & model, const discrete_solution & solution) const
    {
        auto solution_value = model(solution, false);
        CHECK(solution_value.is_feasible());
    }
};

struct CheckOptimality
{
    static std::string name(){ return "CheckOptimality"; }
    void check(const DiscreteGm & model, const discrete_solution & solution) const
    {
        SolutionValue optimal_solution_value;
        discrete_solution optimal_solution;
        std::tie(optimal_solution, optimal_solution_value)  = solve_brute_force(model);
        auto solution_value = model.evaluate(solution, false);

        CHECK(optimal_solution_value.is_feasible() == solution_value.is_feasible());
        if(optimal_solution_value.is_feasible()){
            

            if(!CHECK(solution_value.energy() == doctest::Approx(optimal_solution_value.energy())))
            {
                std::cout<<"optimal solution: "<<std::endl;
                for(auto l : optimal_solution){
                    std::cout<<l<<" ";
                }
                std::cout<<std::endl;
                std::cout<<"        solution: "<<std::endl;
                for(auto l : solution){
                    std::cout<<l<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }
};

struct CheckLocalOptimality
{
    static std::string name(){ return "CheckLocalOptimality"; }
    void check(const DiscreteGm & model, const discrete_solution & solution) const
    {
        auto solution_value = model(solution, false);
        auto solution_copy = solution;

        const auto num_var = model.space().size();
        for(std::size_t vi = 0; vi < num_var; ++vi){
            const auto l = solution[vi];
            const auto num_labels = model.space()[vi];            
            for(discrete_label_type li = 0; li < num_labels; ++li){
                if(li == l){
                     continue;
                }
                solution_copy[vi] = li;
                auto solution_copy_value = model.evaluate(solution_copy, false);
                CHECK(solution_value <= solution_copy_value);
                // reset solution_copy
                solution_copy[vi] = l;
            }
        }
    }
};


template<
    class SOLVER_TYPE,
    class MODEL_GEN_TUPLE,
    class CHECKER_TUPLE
> void test_discrete_gm_optimizer(
    const std::string & testname,
    std::initializer_list<typename SOLVER_TYPE::parameter_type> solver_parameters,
    MODEL_GEN_TUPLE&& model_gen_tuple, 
    std::size_t n_runs,
    CHECKER_TUPLE&& checker_tuple
){
    
    const std::size_t workload = n_runs * solver_parameters.size() * std::tuple_size_v<MODEL_GEN_TUPLE>;

    std::cout<<testname<<":\n";
    progressbar progressBar(workload);
    nxtgm::tuple_for_each(model_gen_tuple, [&](auto && model_gen){

        for(auto i=0; i<n_runs; ++i){

            auto gen_result = model_gen();
            const DiscreteGm model = std::move(gen_result.first);
            const auto name = std::move(gen_result.second);
            //INFO("Model Instance ",name);
            
            auto pi = 0;
            for(auto && solver_parameter : solver_parameters)
            {
                //INFO("Paramter Index", pi);
                auto solver = std::make_unique<SOLVER_TYPE>(model, solver_parameter);

                solver->optimize();
                const auto best_solution = solver->best_solution();

                nxtgm::tuple_for_each(checker_tuple, [&model, &best_solution](auto && checker){
                    // name of the type of the checker
                    using checker_type = std::decay_t<decltype(checker)>;
                    //INFO("Checker ", checker_type::name());

                    checker.check(model, best_solution);
                });
                ++pi;
                progressBar.update();
            }
        }
    });
    std::cout<<"\n";
}

} // namespace nxtgm::tests