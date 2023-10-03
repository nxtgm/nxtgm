#ifndef NXTGM_SPACES_SOLUTION_HPP
#define NXTGM_SPACES_SOLUTION_HPP

namespace nxtgm::space
{
template <class SPACE_TYPE>
class Solution : public std::vector<typename SPACE_TYPE::label_type>
{
  public:
    using base_type = std::vector<typename SPACE_TYPE::label_type>;
    using space_type = SPACE_TYPE;

    Solution(const space_type &space)
        : base_type(space.size(), 0)
    {
    }
};

} // namespace nxtgm::space

#endif // NXTGM_SPACES_SOLUTION_HPP
