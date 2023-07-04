#include <nxtgm/optimizers/gm/discrete/icm.hpp>
#include <nxtgm/utils/timer.hpp>

#include <nxtgm/nxtgm.hpp>

namespace nxtgm {

Icm::Icm(const DiscreteGm &gm, const parameters_type &parameters,
         const solution_type &initial_solution)
    : base_type(gm), parameters_(parameters), movemaker_(gm, initial_solution),
      in_queue_(gm.num_variables(), 1) {
  // put all variables on queue
  for (std::size_t i = 0; i < gm.num_variables(); ++i) {
    queue_.push(i);
  }
}

OptimizationStatus
Icm::optimize(reporter_callback_wrapper_type &reporter_callback,
              repair_callback_wrapper_type & /*repair_callback not used*/
) {
  // indicate the start of the optimization
  reporter_callback.begin();

  // start the timer
  AutoStartedTimer timer;

  // shortcut to the model
  const auto &gm = this->model();

  while (!queue_.empty()) {

    // get next variable
    const auto vi = queue_.front();
    queue_.pop();
    in_queue_[vi] = 0;

    // move optimal
    const auto did_improve = movemaker_.move_optimal(vi);

    // if the solution improved we put all neighbors on the queue
    if (did_improve) {
      // report the current solution to callack
      if (reporter_callback &&
          !timer.paused_call([&]() { return reporter_callback.report(); })) {
        return OptimizationStatus::CALLBACK_EXIT;
      }

      // add all neighbors to the queue
      for (const auto &fi : movemaker_.factors_of_variables()[vi]) {
        for (const auto &neighbour_vi : gm.factors()[fi].variables()) {
          if (neighbour_vi != vi && in_queue_[neighbour_vi] == 0) {
            queue_.push(neighbour_vi);
            in_queue_[neighbour_vi] = 1;
          }
        }
      }
      for (const auto &fi : movemaker_.constraints_of_variables()[vi]) {
        for (const auto &neighbour_vi : gm.constraints()[fi].variables()) {
          if (neighbour_vi != vi && in_queue_[neighbour_vi] == 0) {
            queue_.push(neighbour_vi);
            in_queue_[neighbour_vi] = 1;
          }
        }
      }
    }

    // check if the time limit is reached
    if (timer.elapsed() > this->parameters_.time_limit) {
      return OptimizationStatus::TIME_LIMIT_REACHED;
    }
  }

  // indicate the end of the optimization
  reporter_callback.end();

  return OptimizationStatus::LOCAL_OPTIMAL;
}

SolutionValue Icm::best_solution_value() const {
  return this->movemaker_.solution_value();
}
SolutionValue Icm::current_solution_value() const {
  return this->movemaker_.solution_value();
}

const typename Icm::solution_type &Icm::best_solution() const {
  return this->movemaker_.solution();
}
const typename Icm::solution_type &Icm::current_solution() const {
  return this->movemaker_.solution();
}

} // namespace nxtgm
