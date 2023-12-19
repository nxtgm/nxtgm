#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

namespace nxtgm
{

namespace py = pybind11;

struct PyConsumerProxy
{
    std::function<ProposalConsumerStatus()> consumer;
    discrete_label_type *proposal;
    discrete_label_type *best;
    std::size_t size;
};

class PyProposalGen : public ProposalGenBase
{
  public:
    PyProposalGen(const DiscreteGm &gm_, py::object py_proposal_gen)
        : gm_(gm_),
          py_proposal_gen_(py_proposal_gen)
    {
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {
        // create py::buffer_info from best
        auto best_view = py::array(gm_.num_variables(), best, py::cast(*this, py::return_value_policy::reference));

        auto proposal_view =
            py::array(gm_.num_variables(), proposal, py::cast(*this, py::return_value_policy::reference));

        PyConsumerProxy consumer_proxy;
        consumer_proxy.consumer = consumer;
        // consumer_proxy.proposal = proposal;
        // consumer_proxy.best = const_cast<discrete_label_type *>(best);
        consumer_proxy.size = gm_.num_variables();

        py_proposal_gen_.attr("generate")(consumer_proxy, best_view, proposal_view);
    }

  private:
    const DiscreteGm &gm_;
    py::object py_proposal_gen_;
};

class PyProposalGenFactory : public ProposalGenFactoryBase
{
  public:
    PyProposalGenFactory(py::object py_proposal_gen_factory)
        : py_proposal_gen_factory_(py_proposal_gen_factory)
    {
    }

    // create an instance of the plugin
    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {

        auto py_gm = py::cast(gm, py::return_value_policy::reference);
        py::object py_proposal_gen = py_proposal_gen_factory_.attr("create")(py_gm);
        return std::make_unique<PyProposalGen>(gm, py_proposal_gen);
    }

    // irrelevant since this is not in the plugin registry
    int priority() const override
    {
        return 0;
    }

    // license of the plugin
    std::string license() const override
    {
        return py_proposal_gen_factory_.attr("license")().cast<std::string>();
    }

    // description of the plugin
    std::string description() const override
    {
        return py_proposal_gen_factory_.attr("description")().cast<std::string>();
    }

  private:
    py::object py_proposal_gen_factory_;
};

void export_proposal_gen(py::module_ &pymodule)
{

    py::enum_<ProposalConsumerStatus>(pymodule, "ProposalConsumerStatus")
        .value("ACCEPTED", ProposalConsumerStatus::ACCEPTED)
        .value("REJECTED", ProposalConsumerStatus::REJECTED)
        .value("EXIT", ProposalConsumerStatus::EXIT)
        .export_values();

    py::class_<PyConsumerProxy>(pymodule, "PyConsumerProxy")
        .def(py::init<>())
        .def("__call__",
             [](PyConsumerProxy &consumer_proxy) -> ProposalConsumerStatus { return consumer_proxy.consumer(); });

    // the base class
    py::class_<ProposalGenFactoryBase, std::shared_ptr<ProposalGenFactoryBase>>(pymodule, "ProposalGenFactoryBase")

        ;

    py::class_<PyProposalGen>(pymodule, "PyProposalGen");

    // factory function returning a base class shared ptr
    pymodule.def("_proposal_gen_factory",
                 [](py::object py_proposal_gen_factory) -> std::shared_ptr<ProposalGenFactoryBase> {
                     std::shared_ptr<ProposalGenFactoryBase> ret =
                         std::make_shared<PyProposalGenFactory>(py_proposal_gen_factory);
                     return ret;
                 });
}

} // namespace nxtgm
