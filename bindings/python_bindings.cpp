#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/GraphWrapper.h"
#include "../include/AOTEngine.h"

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

    py::register_exception<HardwareMismatch>(m, "HardwareMismatch");

    py::class_<AOTEngine>(m, "AOTEngine")
        .def(py::init<>())
        .def("compile_ahead_of_time", &AOTEngine::compile_ahead_of_time, "Compile and autotune model to a .kin file",
             py::arg("output_filepath"), py::arg("stream_ptr"))
        .def("load_model", &AOTEngine::load_model, "Load a compiled .kin model",
             py::arg("filepath"));

    py::class_<Serializer>(m, "Serializer")
        .def(py::init<>())
        .def("save_kin_file", &Serializer::save_kin_file, "Save .kin file",
             py::arg("filepath"), py::arg("device_id"), py::arg("weights_hash"), py::arg("op_graph_data"), py::arg("kernel_binaries"))
        .def("load_kin_file", &Serializer::load_kin_file, "Load .kin file and return kernel binaries",
             py::arg("filepath"));
}
