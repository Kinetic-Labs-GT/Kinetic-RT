#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include "../include/GraphWrapper.h"
#include "../include/AOTEngine.h"
#include "../include/Communicator.h"
#include "../include/Router.h"
#include "../include/AsyncAPI.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;


enum class DataType : int {
    UNKNOWN  = 0,
    FLOAT32  = 1,
    FLOAT16  = 2,
    BFLOAT16 = 3,
    INT32    = 4,
    INT64    = 5,
    INT8     = 6,
    BOOL     = 7,
};

static const char* data_type_name(DataType t) {
    switch (t) {
        case DataType::FLOAT32:  return "float32";
        case DataType::FLOAT16:  return "float16";
        case DataType::BFLOAT16: return "bfloat16";
        case DataType::INT32:    return "int32";
        case DataType::INT64:    return "int64";
        case DataType::INT8:     return "int8";
        case DataType::BOOL:     return "bool";
        case DataType::UNKNOWN:
        default:                 return "unknown";
    }
}

static size_t data_type_element_size(DataType t) {
    switch (t) {
        case DataType::FLOAT32:  return 4;
        case DataType::FLOAT16:  return 2;
        case DataType::BFLOAT16: return 2;
        case DataType::INT32:    return 4;
        case DataType::INT64:    return 8;
        case DataType::INT8:     return 1;
        case DataType::BOOL:     return 1;
        case DataType::UNKNOWN:
        default:                 return 0;
    }
}


struct TensorDescriptor {
    uintptr_t data_ptr = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DataType dtype = DataType::UNKNOWN;
    size_t byte_size = 0;  // total bytes spanned by the dense layout
};


static size_t descriptor_numel(const TensorDescriptor& d) {
    if (d.shape.empty()) return 1;
    size_t n = 1;
    for (int64_t s : d.shape) {
        if (s <= 0) return 0;
        const size_t dim = static_cast<size_t>(s);
        if (n > static_cast<size_t>(-1) / dim) {
            throw std::invalid_argument("TensorDescriptor numel overflows size_t.");
        }
        n *= dim;
    }
    return n;
}


static bool descriptor_is_contiguous(const TensorDescriptor& d) {
    if (d.shape.empty()) {
        // 0-d scalar: trivially contiguous, no strides required.
        return true;
    }
    if (d.strides.size() != d.shape.size()) {
        return false;
    }
    const size_t ndim = d.shape.size();
    int64_t expected = 1;
    for (size_t i = ndim; i-- > 0; ) {
        if (d.shape[i] != 1 && d.strides[i] != expected) {
            return false;
        }
        if (expected > (INT64_MAX / d.shape[i])) {
            return false; // Prevent overflow and drop unsafe layout configurations
        }
        expected *= d.shape[i];
    }
    return true;
}


static void validate_tensor_descriptor(const TensorDescriptor& d, const char* name) {
    if (d.data_ptr == 0) {
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') has a null data_ptr; "
            "the underlying storage has not been allocated.");
    }
    if (d.dtype == DataType::UNKNOWN) {
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') has dtype=UNKNOWN; "
            "the Python side failed to map the torch dtype to a supported "
            "DataType enum value.");
    }
    if (d.shape.empty()) {
        // 0-d scalar: allowed. byte_size must equal element_size.
        size_t expected_bytes = data_type_element_size(d.dtype);
        if (d.byte_size != expected_bytes) {
            throw std::invalid_argument(
                std::string("TensorDescriptor('") + name + "') (0-d scalar) "
                "byte_size=" + std::to_string(d.byte_size) +
                " does not match element_size=" + std::to_string(expected_bytes) +
                " for " + data_type_name(d.dtype) + ".");
        }
        return;
    }
    if (d.strides.size() != d.shape.size()) {
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') strides rank (" +
            std::to_string(d.strides.size()) + ") != shape rank (" +
            std::to_string(d.shape.size()) + ").");
    }
    size_t numel = descriptor_numel(d);
    if (numel == 0) {
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') has zero numel; "
            "empty tensors cannot be forwarded across the zero-copy boundary.");
    }
    size_t expected_bytes = numel * data_type_element_size(d.dtype);
    if (d.byte_size != expected_bytes) {
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') byte_size=" +
            std::to_string(d.byte_size) +
            " does not match numel*element_size=" + std::to_string(expected_bytes) +
            " (" + std::to_string(numel) + " * " +
            std::to_string(data_type_element_size(d.dtype)) + " for " +
            data_type_name(d.dtype) + ").");
    }
    if (!descriptor_is_contiguous(d)) {
        std::string shape_str = "[";
        for (size_t i = 0; i < d.shape.size(); ++i) {
            if (i) shape_str += ",";
            shape_str += std::to_string(d.shape[i]);
        }
        shape_str += "]";
        std::string stride_str = "[";
        for (size_t i = 0; i < d.strides.size(); ++i) {
            if (i) stride_str += ",";
            stride_str += std::to_string(d.strides[i]);
        }
        stride_str += "]";
        throw std::invalid_argument(
            std::string("TensorDescriptor('") + name + "') is not C-contiguous: "
            "shape=" + shape_str + " strides=" + stride_str + ". "
            "Sliced or transposed tensors with non-unit strides are explicitly "
            "blocked; call tensor.contiguous() before invoking the runtime.");
    }
}


static TensorDescriptor make_descriptor_from_tensor(py::object tensor, const char* name) {
    TensorDescriptor d;
    d.data_ptr = tensor.attr("data_ptr")().cast<uintptr_t>();
    d.shape    = tensor.attr("shape").cast<std::vector<int64_t>>();
    d.strides  = tensor.attr("stride")().cast<std::vector<int64_t>>();

    int64_t storage_offset = tensor.attr("storage_offset")().cast<int64_t>();
    if (storage_offset != 0) {
        throw std::invalid_argument(
            std::string("torch.Tensor('") + name + "') has storage_offset=" +
            std::to_string(storage_offset) + " (elements). The zero-copy "
            "boundary requires storage_offset == 0; call tensor.clone() (or tensor.contiguous().clone()) "
            "to materialize a dense tensor before invoking the runtime.");
    }

    py::object dtype_obj = tensor.attr("dtype");
    py::module torch_mod = py::module::import("torch");
    if      (dtype_obj.equal(torch_mod.attr("float32")))  d.dtype = DataType::FLOAT32;
    else if (dtype_obj.equal(torch_mod.attr("float16")))  d.dtype = DataType::FLOAT16;
    else if (dtype_obj.equal(torch_mod.attr("bfloat16"))) d.dtype = DataType::BFLOAT16;
    else if (dtype_obj.equal(torch_mod.attr("int32")))    d.dtype = DataType::INT32;
    else if (dtype_obj.equal(torch_mod.attr("int64")))    d.dtype = DataType::INT64;
    else if (dtype_obj.equal(torch_mod.attr("int8")))     d.dtype = DataType::INT8;
    else if (dtype_obj.equal(torch_mod.attr("bool")))     d.dtype = DataType::BOOL;
    else {
        throw std::invalid_argument(
            std::string("torch.Tensor('") + name + "') has unsupported dtype '" +
            std::string(py::str(dtype_obj)) + "'. Supported: float32, float16, "
            "bfloat16, int32, int64, int8, bool.");
    }

    size_t numel = tensor.attr("numel")().cast<size_t>();
    d.byte_size = numel * data_type_element_size(d.dtype);

    size_t untyped_bytes = tensor.attr("untyped_storage")().attr("nbytes")().cast<size_t>();
    if (untyped_bytes < d.byte_size) {
        throw std::invalid_argument(
            std::string("torch.Tensor('") + name + "') logical view exceeds actual physical storage allocation bounds.");
    }

    validate_tensor_descriptor(d, name);
    return d;
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Kinetic-RT (ROCm Runtime) GraphWrapper";

    py::enum_<DataType>(m, "DataType")
        .value("UNKNOWN",  DataType::UNKNOWN)
        .value("FLOAT32",  DataType::FLOAT32)
        .value("FLOAT16",  DataType::FLOAT16)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("INT32",    DataType::INT32)
        .value("INT64",    DataType::INT64)
        .value("INT8",     DataType::INT8)
        .value("BOOL",     DataType::BOOL)
        .export_values();

    py::class_<TensorDescriptor>(m, "TensorDescriptor",
        "Structured descriptor carrying the metadata required to safely "
        "dereference a tensor's underlying storage from native code. "
        "Strides are in *elements* (matching PyTorch semantics).")
        .def(py::init<>())
        .def_readwrite("data_ptr",  &TensorDescriptor::data_ptr)
        .def_readwrite("shape",     &TensorDescriptor::shape)
        .def_readwrite("strides",   &TensorDescriptor::strides)
        .def_readwrite("dtype",     &TensorDescriptor::dtype)
        .def_readwrite("byte_size", &TensorDescriptor::byte_size)
        .def_static("from_tensor",
            [](py::object tensor, const std::string& name) {
                return make_descriptor_from_tensor(tensor, name.c_str());
            },
            "Construct and validate a TensorDescriptor from a live torch.Tensor.",
            py::arg("tensor"), py::arg("name") = "tensor")
        .def("validate",
            [](const TensorDescriptor& self, const std::string& name) {
                validate_tensor_descriptor(self, name.c_str());
            },
            "Re-validate the descriptor's structural invariants.",
            py::arg("name") = "tensor")
        .def("__repr__",
            [](const TensorDescriptor& self) {
                std::string s = "TensorDescriptor(data_ptr=0x";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lx",
                              static_cast<unsigned long>(self.data_ptr));
                s += buf;
                s += ", shape=[";
                for (size_t i = 0; i < self.shape.size(); ++i) {
                    if (i) s += ",";
                    s += std::to_string(self.shape[i]);
                }
                s += "], strides=[";
                for (size_t i = 0; i < self.strides.size(); ++i) {
                    if (i) s += ",";
                    s += std::to_string(self.strides[i]);
                }
                s += "], dtype=";
                s += data_type_name(self.dtype);
                s += ", byte_size=";
                s += std::to_string(self.byte_size);
                s += ")";
                return s;
            });

    py::class_<GraphWrapper>(m, "GraphWrapper")
        .def(py::init<>())
        .def("begin_capture", [](GraphWrapper& self, py::object stream_obj, int batch_size, int seq_len) {
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

    py::class_<KernelLaunchDescriptor>(m, "KernelLaunchDescriptor")
        .def(py::init<>())
        .def_readwrite("input_ptr", &KernelLaunchDescriptor::input_ptr)
        .def_readwrite("output_ptr", &KernelLaunchDescriptor::output_ptr)
        .def_readwrite("qkv_weight_ptr", &KernelLaunchDescriptor::qkv_weight_ptr)
        .def_readwrite("qkv_bias_ptr", &KernelLaunchDescriptor::qkv_bias_ptr)
        .def_readwrite("rms_weight_ptr", &KernelLaunchDescriptor::rms_weight_ptr)
        .def_readwrite("freqs_cos_ptr", &KernelLaunchDescriptor::freqs_cos_ptr)
        .def_readwrite("freqs_sin_ptr", &KernelLaunchDescriptor::freqs_sin_ptr)
        .def_readwrite("k_output_ptr", &KernelLaunchDescriptor::k_output_ptr)
        .def_readwrite("v_output_ptr", &KernelLaunchDescriptor::v_output_ptr)
        .def_readwrite("seq_len", &KernelLaunchDescriptor::seq_len)
        .def_readwrite("d_model", &KernelLaunchDescriptor::d_model)
        .def_readwrite("n_heads", &KernelLaunchDescriptor::n_heads)
        .def_readwrite("head_dim", &KernelLaunchDescriptor::head_dim)
        .def_readwrite("eps", &KernelLaunchDescriptor::eps)
        .def_readwrite("stride_x_seq", &KernelLaunchDescriptor::stride_x_seq)
        .def_readwrite("stride_x_dim", &KernelLaunchDescriptor::stride_x_dim)
        .def_readwrite("stride_w_out", &KernelLaunchDescriptor::stride_w_out)
        .def_readwrite("stride_w_in", &KernelLaunchDescriptor::stride_w_in)
        .def_readwrite("stride_q_seq", &KernelLaunchDescriptor::stride_q_seq)
        .def_readwrite("stride_q_dim", &KernelLaunchDescriptor::stride_q_dim)
        .def_readwrite("stride_k_seq", &KernelLaunchDescriptor::stride_k_seq)
        .def_readwrite("stride_k_dim", &KernelLaunchDescriptor::stride_k_dim)
        .def_readwrite("stride_v_seq", &KernelLaunchDescriptor::stride_v_seq)
        .def_readwrite("stride_v_dim", &KernelLaunchDescriptor::stride_v_dim)
        .def_readwrite("byte_size", &KernelLaunchDescriptor::byte_size)
        .def_readwrite("kernel_name", &KernelLaunchDescriptor::kernel_name)
        .def_readwrite("grid_x", &KernelLaunchDescriptor::grid_x)
        .def_readwrite("grid_y", &KernelLaunchDescriptor::grid_y)
        .def_readwrite("grid_z", &KernelLaunchDescriptor::grid_z)
        .def_readwrite("block_x", &KernelLaunchDescriptor::block_x)
        .def_readwrite("block_y", &KernelLaunchDescriptor::block_y)
        .def_readwrite("block_z", &KernelLaunchDescriptor::block_z)
        .def_readwrite("shared_mem_bytes", &KernelLaunchDescriptor::shared_mem_bytes);

    py::class_<AOTEngine>(m, "AOTEngine")
        .def(py::init<>())
        .def("compile_ahead_of_time", [](AOTEngine& self, const std::string& output_filepath, py::object stream_obj, const std::string& kinetic_target) {
            self.compile_ahead_of_time(output_filepath, py::cast<uintptr_t>(stream_obj), kinetic_target);
        }, "Compile and autotune model to a .kin file",
             py::arg("output_filepath"), py::arg("stream_obj"), py::arg("kinetic_target"))
        .def("load_model", &AOTEngine::load_model, "Load a compiled .kin model",
             py::arg("filepath"))
        .def("launch", [](AOTEngine& self, const TensorDescriptor& input, const TensorDescriptor& output, int seq_len, py::object stream_obj) {
            validate_tensor_descriptor(input,  "AOTEngine.launch.input");
            validate_tensor_descriptor(output, "AOTEngine.launch.output");
            self.launch(reinterpret_cast<void*>(input.data_ptr),
                        reinterpret_cast<void*>(output.data_ptr),
                        seq_len,
                        py::cast<uintptr_t>(stream_obj),
                        input.byte_size);
        }, "Launch AOT kernel with validated TensorDescriptors",
             py::arg("input"), py::arg("output"), py::arg("seq_len"),
             py::arg("stream_obj") = py::int_(0))
        .def("launch_tensor", [](AOTEngine& self, py::object input, py::object output, int seq_len, py::object stream_obj) {
            TensorDescriptor in_d  = make_descriptor_from_tensor(input,  "AOTEngine.launch_tensor.input");
            TensorDescriptor out_d = make_descriptor_from_tensor(output, "AOTEngine.launch_tensor.output");
            self.launch(reinterpret_cast<void*>(in_d.data_ptr),
                        reinterpret_cast<void*>(out_d.data_ptr),
                        seq_len,
                        py::cast<uintptr_t>(stream_obj),
                        in_d.byte_size);
        }, "Launch AOT kernel with live torch.Tensor objects (validates layout internally)",
             py::arg("input"), py::arg("output"), py::arg("seq_len"),
             py::arg("stream_obj") = py::int_(0))
        .def("launch", [](AOTEngine& self, uintptr_t input_ptr, uintptr_t output_ptr, int seq_len, py::object stream_obj, size_t byte_size) {
            self.launch(reinterpret_cast<void*>(input_ptr), reinterpret_cast<void*>(output_ptr), seq_len, py::cast<uintptr_t>(stream_obj), byte_size);
        }, "Launch AOT kernel asynchronously with explicit input/output buffers",
             py::arg("input_ptr"), py::arg("output_ptr"), py::arg("seq_len"), py::arg("stream_obj") = py::int_(0), py::arg("byte_size") = 0)
        .def("launch_descriptor", [](AOTEngine& self, KernelLaunchDescriptor& descriptor, py::object stream_obj) {
            self.launch(descriptor, py::cast<uintptr_t>(stream_obj));
        }, "Launch AOT kernel asynchronously with a structured descriptor",
             py::arg("descriptor"), py::arg("stream_obj"))
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
        .def("launch", [](HardwareRouter& self, const TensorDescriptor& input, const TensorDescriptor& output, int seq_len, size_t byte_size) {
            validate_tensor_descriptor(input,  "HardwareRouter.launch.input");
            validate_tensor_descriptor(output, "HardwareRouter.launch.output");
            size_t effective_byte_size = (byte_size != 0) ? byte_size : input.byte_size;
            self.launch(reinterpret_cast<void*>(input.data_ptr),
                        reinterpret_cast<void*>(output.data_ptr),
                        seq_len,
                        effective_byte_size);
        }, "Launch inference through the hardware-aware router via validated TensorDescriptors",
             py::arg("input"), py::arg("output"), py::arg("seq_len"), py::arg("byte_size") = 0)
        .def("launch_tensor", [](HardwareRouter& self, py::object input, py::object output, int seq_len, size_t byte_size) {
            TensorDescriptor in_d  = make_descriptor_from_tensor(input,  "HardwareRouter.launch_tensor.input");
            TensorDescriptor out_d = make_descriptor_from_tensor(output, "HardwareRouter.launch_tensor.output");
            size_t effective_byte_size = (byte_size != 0) ? byte_size : in_d.byte_size;
            self.launch(reinterpret_cast<void*>(in_d.data_ptr),
                        reinterpret_cast<void*>(out_d.data_ptr),
                        seq_len,
                        effective_byte_size);
        }, "Launch inference with live torch.Tensor objects (validates layout internally)",
             py::arg("input"), py::arg("output"), py::arg("seq_len"), py::arg("byte_size") = 0)
        .def("launch", [](HardwareRouter& self, uintptr_t input_ptr, uintptr_t output_ptr, int seq_len, size_t byte_size) {
            self.launch(reinterpret_cast<void*>(input_ptr), reinterpret_cast<void*>(output_ptr), seq_len, byte_size);
        }, "Launch inference through the hardware-aware router via pointers",
             py::arg("input_ptr"), py::arg("output_ptr"), py::arg("seq_len"), py::arg("byte_size") = 0);

    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("submit", [](InferenceQueue& self, const TensorDescriptor& input, int max_tokens, const std::string& request_id) {
            validate_tensor_descriptor(input, "InferenceQueue.submit.input");
            size_t input_len = descriptor_numel(input);
            self.submit(input.data_ptr, input_len, max_tokens, request_id);
        }, "Submit a request using a validated TensorDescriptor (input_len derived from shape)",
             py::arg("input"), py::arg("max_tokens"), py::arg("request_id"))
        .def("submit_tensor", [](InferenceQueue& self, py::object input, int max_tokens, const std::string& request_id) {
            TensorDescriptor in_d = make_descriptor_from_tensor(input, "InferenceQueue.submit_tensor.input");
            size_t input_len = descriptor_numel(in_d);
            self.submit(in_d.data_ptr, input_len, max_tokens, request_id);
        }, "Submit a request using a live torch.Tensor (validates layout internally)",
             py::arg("input"), py::arg("max_tokens"), py::arg("request_id"))
        .def("submit", [](InferenceQueue& self, uintptr_t input_ptr, size_t input_len, int max_tokens, const std::string& request_id) {
            self.submit(input_ptr, input_len, max_tokens, request_id);
        }, "Submit a request to the C++ engine natively",
             py::arg("input_ptr"), py::arg("input_len"), py::arg("max_tokens"), py::arg("request_id"))
        .def("poll", &InferenceQueue::poll, "Poll for stream responses via lock-free queue");

    py::class_<InferenceWorker>(m, "InferenceWorker")
        .def(py::init<InferenceQueue&, HardwareRouter&>(), py::arg("queue"), py::arg("router"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), "Initialize background C++ task consumer executing hardware routines");
}
