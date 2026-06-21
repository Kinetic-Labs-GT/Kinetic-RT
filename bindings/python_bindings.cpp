#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/GraphWrapper.h"
#include "../include/AOTEngine.h"
#include "../include/Communicator.h"
#include "../include/Router.h"
#include "../include/AsyncAPI.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Kinetic-RT (ROCm Runtime) GraphWrapper";

    py::class_<GraphWrapper>(m, "GraphWrapper")
        .def(py::init<>())
        .def("begin_capture", [](GraphWrapper& self, py::object stream_obj, int batch_size, int seq_len) {
            // Keep reference to python object
            self.begin_capture(py::cast<uintptr_t>(stream_obj), batch_size, seq_len);
        }, "Begin capturing operations into a graph on the given stream",
             py::arg("stream_obj"), py::arg("batch_size"), py::arg("seq_len"))
        .def("end_capture", [](GraphWrapper& self, py::object stream_obj) {
            self.end_capture(py::cast<uintptr_t>(stream_obj));
        }, "End graph capture and instantiate",
             py::arg("stream_obj"))
        .def("launch", [](GraphWrapper& self, std::vector<py::object> stream_objs, std::vector<py::object> buffers) {
            self.launch(stream_objs, buffers);
        }, "Launch the instantiated graph concurrently, holding buffer references",
             py::arg("stream_objs"), py::arg("buffers") = std::vector<py::object>())
        .def("is_valid", &GraphWrapper::is_valid, "Check if graph is valid for current batch size and sequence length",
             py::arg("batch_size"), py::arg("seq_len"))
        .def("invalidate", &GraphWrapper::invalidate, "Manually invalidate the graph");

    py::register_exception<HardwareMismatchError>(m, "HardwareMismatchError");

    py::class_<AOTEngine>(m, "AOTEngine")
        .def(py::init<>())
        .def("compile_ahead_of_time", [](AOTEngine& self, const std::string& output_filepath, py::object stream_obj, const std::string& kinetic_target) {
            self.compile_ahead_of_time(output_filepath, py::cast<uintptr_t>(stream_obj), kinetic_target);
        }, "Compile and autotune model to a .kin file",
             py::arg("output_filepath"), py::arg("stream_obj"), py::arg("kinetic_target"))
        .def("load_model", &AOTEngine::load_model, "Load a compiled .kin model",
             py::arg("filepath"))
        .def("launch", [](AOTEngine& self, py::object py_input, py::object stream_obj, size_t byte_size) {
            self.launch(py_input, py::cast<uintptr_t>(stream_obj), byte_size);
        }, "Launch AOT kernel asynchronously with pinned buffers",
             py::arg("py_input"), py::arg("stream_obj"), py::arg("byte_size") = 0)
        .def("synchronize_and_clear", [](AOTEngine& self, py::object stream_obj) {
            self.synchronize_and_clear(py::cast<uintptr_t>(stream_obj));
        }, "Synchronize stream and clear pinned buffers",
             py::arg("stream_obj"));

    py::class_<Serializer>(m, "Serializer")
        .def(py::init<>())
        .def("save_kin_file", &Serializer::save_kin_file, "Save .kin file",
             py::arg("filepath"), py::arg("device_id"), py::arg("kinetic_target"), py::arg("weights_hash"), py::arg("op_graph_data"), py::arg("kernel_binaries"))
        .def("load_kin_file", &Serializer::load_kin_file, "Load .kin file and return kernel binaries",
             py::arg("filepath"))
        .def("get_tensor_parallel_degree", &Serializer::get_tensor_parallel_degree, "Get tensor parallel degree from metadata",
             py::arg("filepath"));

    py::class_<Communicator>(m, "Communicator")
        .def(py::init<int, int>(), py::arg("rank"), py::arg("world_size"))
        .def("all_reduce_async", [](Communicator& self, uintptr_t sendbuff, uintptr_t recvbuff, size_t count, int datatype, int op, py::object stream_obj) {
            uintptr_t stream_ptr = py::cast<uintptr_t>(stream_obj);
            py::gil_scoped_release release;
            self.all_reduce_async(reinterpret_cast<void*>(sendbuff), reinterpret_cast<void*>(recvbuff), count, datatype, op, stream_ptr);
        }, "Perform async all_reduce", py::arg("sendbuff"), py::arg("recvbuff"), py::arg("count"), py::arg("datatype"), py::arg("op"), py::arg("stream_obj"));

    py::class_<HardwareRouter>(m, "HardwareRouter")
        .def(py::init<>())
        .def("load_model", &HardwareRouter::load_model, "Load a compiled .kin model or TensorRT plan", py::arg("filepath"))
        .def("launch", [](HardwareRouter& self, uintptr_t input_ptr, uintptr_t output_ptr, int seq_len, size_t byte_size) {
            self.launch(reinterpret_cast<void*>(input_ptr), reinterpret_cast<void*>(output_ptr), seq_len, byte_size);
        }, "Launch inference through the hardware-aware router via pointers", py::arg("input_ptr"), py::arg("output_ptr"), py::arg("seq_len"), py::arg("byte_size") = 0);

    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("submit", [](InferenceQueue& self, uintptr_t input_ptr, size_t input_len, int max_tokens, const std::string& request_id) {
            self.submit(input_ptr, input_len, max_tokens, request_id);
        }, "Submit a request to the C++ engine natively",
             py::arg("input_ptr"), py::arg("input_len"), py::arg("max_tokens"), py::arg("request_id"))
        .def("poll", &InferenceQueue::poll, "Poll for stream responses via lock-free queue");

    py::class_<InferenceWorker>(m, "InferenceWorker")
        .def(py::init<InferenceQueue&, HardwareRouter&>(), py::arg("queue"), py::arg("router"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), "Initialize background C++ task consumer executing hardware routines");
}