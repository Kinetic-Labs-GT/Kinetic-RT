import time
import python.kinetic_rt as kinetic_rt

def benchmark():
    wrapper = kinetic_rt.GraphWrapper()
    stream_ptr = 1234

    wrapper.begin_capture(stream_ptr, batch_size=1, seq_len=128)
    wrapper.end_capture(stream_ptr)

    # 100 buffers
    buffers = [i for i in range(100)]

    start_time = time.time()
    for _ in range(100000):
        wrapper.launch([stream_ptr], buffers)
    end_time = time.time()

    wrapper.invalidate()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
