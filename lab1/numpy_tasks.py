"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros_like(np.asarray(vectors[0]))
    for matrix, vector in zip(matrices, vectors):
        result += matrix @ vector
    return result

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for row in matrix:
        result.append(np.unique(row).tolist())
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for column in np.asarray(matrix).T:
        result.append(np.unique(column).tolist())
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns = data.rows, data.columns
    mean, std, seed = data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))
    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


def plot_matrix_histograms(matrix: np.ndarray) -> None:
    """Строит гистограммы значений строк и столбцов."""
    matrix = np.asarray(matrix)
    for i in range(matrix.shape[0]):
        plt.figure()
        plt.hist(matrix[i])
        plt.title(f"Строка {i}")
        plt.show()
    for i in range(matrix.shape[1]):
        plt.figure()
        plt.hist(matrix[:, i])
        plt.title(f"Столбец {i}")
        plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = data.rows, data.columns
    first, second = data.first, data.second
    result = np.zeros((rows, columns), dtype=np.result_type(first, second))
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                result[i, j] = first
            else:
                result[i, j] = second
    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    x_start = (image_width - width) // 2
    y_start = (image_height - height) // 2
    x_end = x_start + width
    y_end = y_start + height
    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    x0 = image_width // 2
    y0 = image_height // 2
    y, x = np.ogrid[:image_height, :image_width]
    ellipse = (
        ((x - x0) / semi_axis_x) ** 2
        + ((y - y0) / semi_axis_y) ** 2
    ) <= 1
    image[ellipse] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values)
    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)
    local_maxima = []
    local_minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)

        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)
    moving_average = []
    for i in range(len(values) - window + 1):
        part = values[i:i + window]
        moving_average.append(np.mean(part))
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=np.array(local_maxima),
        local_minima_indices=np.array(local_minima),
        moving_average=np.array(moving_average),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = np.max(labels) + 1
    result = np.zeros((len(labels), class_count), dtype=int)
    for i in range(len(labels)):
        result[i, labels[i]] = 1

    return result
