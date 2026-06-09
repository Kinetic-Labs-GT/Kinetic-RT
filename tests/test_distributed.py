import python.kinetic_rt as kinetic_rt
import ctypes
import threading
import logging

logger = logging.getLogger(__name__)

def test_distributed_sharding():
    world_size = 2
    comm0 = kinetic_rt.Communicator(0, world_size)
    comm1 = kinetic_rt.Communicator(1, world_size)

    count = 100

    # We use ctypes arrays to simulate the buffers since mock_ncclAllReduce takes raw pointers.
    FloatArrayType = ctypes.c_float * count

    tensor0 = FloatArrayType(*([1.0] * count))
    tensor1 = FloatArrayType(*([2.0] * count))

    recv0 = FloatArrayType(*([0.0] * count))
    recv1 = FloatArrayType(*([0.0] * count))

    gw = kinetic_rt.GraphWrapper()

    stream0_ptr = 0x1000
    stream1_ptr = 0x2000

    gw.begin_capture(stream0_ptr, 1, 1)
    # the end_capture just instantiates, we need to launch

    t1 = threading.Thread(target=lambda: comm0.all_reduce_async(ctypes.addressof(tensor0), ctypes.addressof(recv0), count, 0, 0, stream0_ptr))
    t2 = threading.Thread(target=lambda: comm1.all_reduce_async(ctypes.addressof(tensor1), ctypes.addressof(recv1), count, 0, 0, stream1_ptr))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    gw.end_capture(stream0_ptr)

    # buffers argument expects a list/vector of python objects to hold references
    gw.launch([stream0_ptr, stream1_ptr], [])

    gw.invalidate()

    for i in range(count):
        assert recv0[i] == 3.0, f"Expected 3.0, got {recv0[i]}"
        assert recv1[i] == 3.0, f"Expected 3.0, got {recv1[i]}"

    logger.info("Mathematical validation tests passed.")

if __name__ == "__main__":
    test_distributed_sharding()
