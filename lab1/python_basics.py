
"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


def count_vowels(data: TextInput) -> int:
    text = data.value
    result = sum(1 for char in text.lower() if char in "aeiou")
    return result


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    result = len(set(text)) == len(text)
    return result


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0

    while number > 9:
        result = 1

        for digit in str(number):
            result *= int(digit)

        number = result
        count += 1

    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")

    if len(predicted) == 0:
        raise ValueError("Vectors must not be empty")

    return sum(
        (x - y) ** 2
        for x, y in zip(predicted, expected)
    ) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            parts.append(f"({divisor})")
        elif power > 1:
            parts.append(f"({divisor}**{power})")

        divisor += 1

    if number > 1:
        parts.append(f"({number})")

    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    count = 0
    result = 0

    while result < cube_count:
        count += 1
        result += count * count

    if result == cube_count:
        return count

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(digit) for digit in str(number)]

    middle = len(digits) // 2

    if len(digits) % 2 == 0:
        left = digits[:middle - 1]
        right = digits[middle + 1:]
    else:
        left = digits[:middle]
        right = digits[middle + 1:]

    return sum(left) == sum(right)
