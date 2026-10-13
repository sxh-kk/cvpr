"""Bounded CPU preparation for DINO, preserving the reference PIL pixels."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
import threading

import numpy as np
from PIL import Image


def optimization_options(config):
    precision = config.get("inference_precision", "fp32")
    preprocessor = config.get("preprocessor", "hf")
    attention = config.get("attention_implementation", "auto")
    if precision not in {"fp32", "tf32", "bf16"}:
        raise ValueError("DINO inference_precision must be fp32, tf32 or bf16")
    if preprocessor not in {"hf", "pil_parallel"}:
        raise ValueError("DINO preprocessor must be hf or pil_parallel")
    if attention not in {"auto", "sdpa", "eager"}:
        raise ValueError("DINO attention_implementation must be auto, sdpa or eager")
    options = {"inference_precision": precision, "preprocessor": preprocessor,
               "attention_implementation": attention}
    for name, default, minimum in [("preprocess_workers", 1, 1), ("prefetch_batches", 0, 0),
                                   ("decoder_threads", 0, 0), ("cpu_threads", 4, 1)]:
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"DINO {name} must be an integer >= {minimum}")
        options[name] = value
    for name in ["pin_memory", "compile_model", "optimized_storage"]:
        value = config.get(name, False)
        if not isinstance(value, bool):
            raise ValueError(f"DINO {name} must be boolean")
        options[name] = value
    return options


class ParallelPILPreprocessor:
    """The HF/UniEgoMotion resize and normalization, with parallel PIL resize."""

    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dino-resize")
        self.mean = np.asarray([.485, .456, .406], dtype=np.float32)
        self.std = np.asarray([.229, .224, .225], dtype=np.float32)

    def image(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Parallel DINO preprocessing requires HWC RGB uint8")
        height, width = image.shape[:2]
        if min(height, width) < 1:
            raise ValueError("DINO image dimensions must be positive")
        size = (int(width*336/height), 336) if height <= width else (336, int(height*336/width))
        pixels = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))
        # HF rescale promotes to float64, then rounds to float32 before normalize.
        pixels = (pixels.astype(np.float64)*(1/255)).astype(np.float32)
        pixels = (pixels-self.mean)/self.std
        return pixels.transpose(2, 0, 1)

    def __call__(self, images):
        if not images:
            raise ValueError("DINO requires a nonempty image batch")
        return np.stack(list(self.pool.map(self.image, images)))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def prepared_batches(iterator, prepare, depth):
    """Prepare on a producer thread; only it advances/closes the source iterator.

    The queue retains at most ``depth`` resized batches. Exceptions reach the
    consumer and early close stops the producer before returning.
    """
    if depth == 0:
        try:
            for images, indices, timestamps in iterator:
                yield prepare(images), indices, timestamps
        finally:
            iterator.close()
        return
    queue = Queue(maxsize=depth)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=.1)
                return True
            except Full:
                pass
        return False

    def produce():
        try:
            for images, indices, timestamps in iterator:
                if stop.is_set() or not put(("batch", (prepare(images), indices, timestamps))):
                    break
        except BaseException as error:
            put(("error", error))
        finally:
            try:
                iterator.close()
            finally:
                put(("end", None))

    worker = threading.Thread(target=produce, name="dino-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, item = queue.get(timeout=.1)
            except Empty:
                if not worker.is_alive():
                    raise RuntimeError("DINO prefetch producer exited without an end marker")
                continue
            if kind == "error":
                raise item
            if kind == "end":
                break
            yield item
    finally:
        stop.set()
        worker.join()
