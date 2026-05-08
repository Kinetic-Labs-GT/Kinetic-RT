#include <pybind11/pybind11.h>
#include "../include/GraphWrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(kinetic_rt, m) {
    m.doc() = "Kinetic-RT (ROCm Runtime) GraphWrapper";

    py::class_<GraphWrapper>(m, "GraphWrapper")
        .def(py::init<>())
        .def("begin_capture", &GraphWrapper::begin_capture, "Begin capturing operations into a graph on the given stream",
             py::arg("stream_ptr"), py::arg("batch_size"), py::arg("seq_len"))
        .def("end_capture", &GraphWrapper::end_capture, "End graph capture and instantiate",
             py::arg("stream_ptr"))
        .def("launch", &GraphWrapper::launch, "Launch the instantiated graph",
             py::arg("stream_ptr"))
        .def("is_valid", &GraphWrapper::is_valid, "Check if graph is valid for current batch size and sequence length",
             py::arg("batch_size"), py::arg("seq_len"))
        .def("invalidate", &GraphWrapper::invalidate, "Manually invalidate the graph");
}
