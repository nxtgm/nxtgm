#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <vector>

namespace nxtgm
{

namespace em = emscripten;

std::string get_exception_message(int exceptionPtr)
{
    auto ptr = reinterpret_cast<std::exception *>(exceptionPtr);

    // get traceback
    std::vector<std::string> traceback;

    return std::string(ptr->what());
}

void export_functions();
void export_space();
void export_optimizer();
void export_callbacks();
void export_gm();
void export_proposal_gen();

EMSCRIPTEN_BINDINGS(my_module)
{

    em::register_vector<discrete_label_type>("VectorUInt16");
    em::function("get_exception_message", &get_exception_message);

    export_space();
    export_gm();
    export_functions();
    export_optimizer();
    export_callbacks();
    export_proposal_gen();
}

} // namespace nxtgm
