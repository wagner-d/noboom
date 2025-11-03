import numpy as np
import numpy.typing as npt


def compute_continuous_segments(array: npt.NDArray[int], value: int) -> list[tuple[int, int]]:

    binary_array = np.where(array == value, 1, 0)

    boundaries = np.zeros_like(binary_array)
    boundaries[1:] = binary_array[:-1]
    boundaries = binary_array - boundaries

    indices = boundaries.nonzero()[0].tolist()

    if len(indices) % 2:
        indices.append(array.shape[0])

    return [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]


def compute_events(array: npt.NDArray[int]) -> list[tuple[int, int, int, int]]:

    return [(segment[0],
             segment[0] + int(np.min(np.where(np.append(array[segment[0]:segment[1]], 3) > 1))),
             segment[0] + int(np.min(np.where(np.append(array[segment[0]:segment[1]], 3) > 2))),
             segment[1]) for segment in compute_continuous_segments((array > 0).astype(int), 1)]


