
import unittest
import numpy as np

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)


class TestPythonBasics(unittest.TestCase):

    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput(value="Hello")), 2)
        self.assertEqual(count_vowels(TextInput(value="AEIOU")), 5)
        self.assertEqual(count_vowels(TextInput(value="rhythm")), 0)
        self.assertEqual(count_vowels(TextInput(value="")), 0)

    def test_has_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput(value="abc")))
        self.assertFalse(has_unique_characters(TextInput(value="hello")))
        self.assertTrue(has_unique_characters(TextInput(value="")))
        self.assertTrue(has_unique_characters(TextInput(value="Aa")))
        self.assertFalse(has_unique_characters(TextInput(value="aa")))

    def test_count_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(value=1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(value=7)), 3)
        self.assertEqual(count_one_bits(PositiveIntegerInput(value=8)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(value=15)), 4)

    def test_multiplicative_persistence(self):
        self.assertEqual(
            multiplicative_persistence(PositiveIntegerInput(value=39)), 3
        )
        self.assertEqual(
            multiplicative_persistence(PositiveIntegerInput(value=4)), 0
        )
        self.assertEqual(
            multiplicative_persistence(PositiveIntegerInput(value=999)), 4
        )
        self.assertEqual(
            multiplicative_persistence(PositiveIntegerInput(value=25)), 2
        )

    def test_mse(self):
        data = VectorPairInput(
            predicted=[1, 2, 3],
            expected=[1, 3, 5],
        )
        self.assertAlmostEqual(mse(data), 5 / 3)

        data = VectorPairInput(
            predicted=[1, 2],
            expected=[1, 2],
        )
        self.assertEqual(mse(data), 0)

        data = VectorPairInput(
            predicted=[0, 0],
            expected=[2, 4],
        )
        self.assertEqual(mse(data), 10)

        with self.assertRaises(ValueError):
            mse(VectorPairInput(predicted=[1], expected=[1, 2]))

        with self.assertRaises(ValueError):
            mse(VectorPairInput(predicted=[], expected=[]))

    def test_prime_factorization(self):
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(value=2)), "(2)"
        )
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(value=12)), "(2**2)(3)"
        )
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(value=49)), "(7**2)"
        )
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(value=86240)),
            "(2**5)(5)(7**2)(11)",
        )
        self.assertEqual(
            prime_factorization(PositiveIntegerInput(value=13)), "(13)"
        )

    def test_pyramid(self):
        self.assertEqual(pyramid(PositiveIntegerInput(value=1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(value=5)), 2)
        self.assertEqual(pyramid(PositiveIntegerInput(value=14)), 3)
        self.assertEqual(pyramid(PositiveIntegerInput(value=30)), 4)
        self.assertEqual(pyramid(PositiveIntegerInput(value=4)), "It is impossible")

    def test_is_balanced_number(self):
        self.assertTrue(
            is_balanced_number(PositiveIntegerInput(value=1234006))
        )
        self.assertFalse(
            is_balanced_number(PositiveIntegerInput(value=123456))
        )
        self.assertFalse(
            is_balanced_number(PositiveIntegerInput(value=123))
        )
        self.assertTrue(
            is_balanced_number(PositiveIntegerInput(value=121))
        )
        self.assertTrue(
            is_balanced_number(PositiveIntegerInput(value=11))
        )


class TestNumPyTasks(unittest.TestCase):

    def test_sum_prod(self):
        matrices = np.array([
            [[1, 2], [3, 4]],
            [[2, 0], [0, 2]],
        ])
        vectors = np.array([
            [[1], [2]],
            [[3], [4]],
        ])

        data = MatrixVectorBatchInput(
            matrices=matrices,
            vectors=vectors,
        )

        expected = np.array([[11], [19]])
        np.testing.assert_allclose(sum_prod(data), expected)

    def test_binarize(self):
        matrix = np.array([[1, 3], [5, 2]])
        original = matrix.copy()

        data = BinarizeInput(matrix=matrix, threshold=2)
        result = binarize(data)

        np.testing.assert_array_equal(
            result,
            np.array([[0, 1], [1, 0]]),
        )
        np.testing.assert_array_equal(matrix, original)

        data = BinarizeInput(
            matrix=np.array([[2, 2], [2, 2]]),
            threshold=2,
        )
        np.testing.assert_array_equal(
            binarize(data),
            np.zeros((2, 2), dtype=int),
        )

    def test_unique_rows(self):
        data = MatrixInput(
            matrix=np.array([
                [1, 2, 2, 3],
                [4, 4, 4, 4],
                [3, 1, 3, 1],
            ])
        )

        self.assertEqual(
            unique_rows(data),
            [[1, 2, 3], [4], [1, 3]],
        )

    def test_unique_columns(self):
        data = MatrixInput(
            matrix=np.array([
                [1, 2, 1],
                [1, 3, 4],
                [2, 2, 4],
            ])
        )

        self.assertEqual(
            unique_columns(data),
            [[1, 2], [2, 3], [1, 4]],
        )

    def test_matrix_statistics(self):
        data = RandomMatrixInput(
            rows=3,
            columns=4,
            mean=0,
            std=1,
            seed=42,
        )

        result = matrix_statistics(data)

        self.assertEqual(result.matrix.shape, (3, 4))

        np.testing.assert_allclose(
            result.row_means,
            np.mean(result.matrix, axis=1),
        )
        np.testing.assert_allclose(
            result.column_means,
            np.mean(result.matrix, axis=0),
        )
        np.testing.assert_allclose(
            result.row_variances,
            np.var(result.matrix, axis=1),
        )
        np.testing.assert_allclose(
            result.column_variances,
            np.var(result.matrix, axis=0),
        )

        result2 = matrix_statistics(data)
        np.testing.assert_array_equal(result.matrix, result2.matrix)

    def test_chess(self):
        data = ChessInput(rows=3, columns=4, first=0, second=1)

        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])

        np.testing.assert_array_equal(chess(data), expected)

        data = ChessInput(rows=1, columns=1, first=5, second=9)
        np.testing.assert_array_equal(chess(data), [[5]])

    def test_draw_rectangle(self):
        data = RectangleInput(
            width=2,
            height=2,
            image_height=4,
            image_width=4,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        result = draw_rectangle(data)

        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(
           result[1:3, 1:3],
            np.full((2, 2, 3), [255, 0, 0])
        )
        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])

    def test_draw_ellipse(self):
        data = EllipseInput(
            semi_axis_x=1,
            semi_axis_y=1,
            image_height=3,
            image_width=3,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        result = draw_ellipse(data)

        self.assertEqual(result.shape, (3, 3, 3))
        np.testing.assert_array_equal(result[1, 1], [255, 0, 0])
        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])

    def test_analyze_time_series(self):
        data = TimeSeriesInput(
            values=[1, 3, 1, 0, 2, 0],
            window=3,
        )

        result = analyze_time_series(data)

        self.assertAlmostEqual(result.mean, 7 / 6)
        self.assertAlmostEqual(result.variance, np.var([1, 3, 1, 0, 2, 0]))
        self.assertAlmostEqual(result.std, np.std([1, 3, 1, 0, 2, 0]))

        self.assertEqual(result.local_maxima_indices.tolist(), [1, 4])
        self.assertEqual(result.local_minima_indices.tolist(), [3])

        np.testing.assert_allclose(
            result.moving_average,
            [5 / 3, 4 / 3, 1, 2 / 3],
        )

    def test_one_hot(self):
        data = OneHotInput(
            labels=[0, 2, 3, 0],
            class_count=None,
        )

        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ])

        np.testing.assert_array_equal(one_hot(data), expected)

    def test_one_hot_explicit_class_count(self):
        data = OneHotInput(
            labels=[0, 1, 0],
            class_count=4,
        )

        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ])

        np.testing.assert_array_equal(one_hot(data), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
