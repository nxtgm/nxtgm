#include "indexed_heap_queue.hpp"

#include <xtensor/xtensor.hpp>

namespace nxtgm
{
class AssigmentQueue
{
  public:
    AssigmentQueue(const xt::xtensor<double, 2> &costs, bool with_ignore_label)
        : queue_(costs.size()),
          with_ignore_label_(with_ignore_label),
          num_var_(costs.shape()[0]),
          num_labels_(costs.shape()[1])
    {
        for (std::size_t i = 0; i < costs.size(); i++)
        {
            queue_.push(i, costs[i]);
        }
    }

    inline bool empty() const
    {
        return queue_.empty();
    }

    std::tuple<std::size_t, std::size_t, double> pop()
    {
        const auto energy = queue_.topPriority();
        queue_.pop();

        auto [var, label] = xt::unravel_index(index, std::array<std::size_t, 2>{num_var_, num_labels_});
        if (!(label == 0 && with_ignore_label_))
        {
            for (std::size_t other_vars = 0; other_vars < num_var_; other_vars++)
            {
                if (other_vars != var)
                {
                    queue_.deleteItem(var * num_labels_ + label);
                }
            }
        }
        return {var, label, energy};
    }

  private:
    IndexedHeapQueue<double> queue_;
    bool with_ignore_label_;
    std::size_t num_var_;
    std::size_t num_labels_;
};
} // namespace nxtgm
