from typing import List
import heapq


def merge_sort_subsets(p: List[int]) -> List[int]:
    """
    This function takes a list of numbers and returns the sorted list of all possible sums of the numbers in the list.
    :param p: list of numbers
    :return: sorted list of all possible sums of the numbers in the list
    """
    n = len(p)
    t = 1 << n

    a = [0] * t
    a[t - 1] = p[0]

    for i in range(1, n):
        i1 = t - (1 << i)
        j1 = i1
        k1 = t - (2 << i)

        while i1 < t and j1 < t:
            if a[i1] <= a[j1] + p[i]:
                a[k1] = a[i1]
                k1 += 1
                i1 += 1
            else:
                a[k1] = a[j1] + p[i]
                k1 += 1
                j1 += 1

        while j1 < t:
            a[j1] += p[i]
            j1 += 1

    return a


def subset_sum_dp(numbers: List[int], target: int) -> List[int]:
    """
    This function takes a list of numbers and a target number and returns a subset of the numbers that add up to the target number using dynamic programming.
    :param numbers: list of numbers
    :param target: number that the subset should add up to
    :return: list with the indexes of the numbers that add up to the target number
    """

    print("This is a terrible solution for big targets, but it works for small ones.")
    print("The time complexity is O(n * target) and the space complexity is O(n * target).")

    n = len(numbers)
    dp = [[False] * (target + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = True

    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if j < numbers[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - numbers[i - 1]]

    if not dp[n][target]:
        return []

    subset = []
    i, j = n, target
    while i > 0 and j > 0:
        if dp[i][j] and not dp[i - 1][j]:
            subset.append(i - 1)
            j -= numbers[i - 1]
        i -= 1

    return subset[::-1]


def horowitz_sahni(p, desired_sum):
    """
    This function takes a list of numbers and a target number and returns a boolean value indicating whether a subset was found using Horowitz-Sahni's algorithm.
    :param p:
    :param desired_sum:
    :return: boolean value indicating whether a subset was found
    """
    n = len(p)

    if n % 2 == 0:
        # If it's even, e.g. 12, silly_index = 6
        silly_index = n // 2
    else:
        # If it's odd, e.g. 13, silly_index = 7
        silly_index = n // 2 + 1

    # Two arrays, a and b: a are the numbers p[0] to p[n/2 - 1] and b are p[n/2] to p[n-1]
    a = p[:silly_index]
    b = p[silly_index:]

    a_sums = [0] * (1 << silly_index)
    b_sums = [0] * (1 << (n - silly_index))

    for i in range(silly_index):
        a_sums[i] = a[i]
    for i in range(n - silly_index):
        b_sums[i] = b[i]

    a_sums[(1 << silly_index) - 2] = 0
    a_sums[(1 << silly_index) - 1] = a[0]
    for i in range(1, silly_index):
        i1 = (1 << silly_index) - (1 << i)
        j1 = i1
        k1 = (1 << silly_index) - (2 << i)
        while i1 < (1 << silly_index) and j1 < (1 << silly_index):
            if a_sums[i1] <= a_sums[j1] + a[i]:
                a_sums[k1] = a_sums[i1]
                k1 += 1
                i1 += 1
            else:
                a_sums[k1] = a_sums[j1] + a[i]
                k1 += 1
                j1 += 1
        while j1 < (1 << silly_index):
            a_sums[j1] += a[i]
            j1 += 1

    b_sums[(1 << silly_index) - 2] = 0
    b_sums[(1 << silly_index) - 1] = b[0]
    for i in range(1, n - silly_index):
        i1 = (1 << (n - silly_index)) - (1 << i)
        j1 = i1
        k1 = (1 << (n - silly_index)) - (2 << i)
        while i1 < (1 << (n - silly_index)) and j1 < (1 << (n - silly_index)):
            if b_sums[i1] <= b_sums[j1] + b[i]:
                b_sums[k1] = b_sums[i1]
                k1 += 1
                i1 += 1
            else:
                b_sums[k1] = b_sums[j1] + b[i]
                k1 += 1
                j1 += 1
        while j1 < (1 << (n - silly_index)):
            b_sums[j1] += b[i]
            j1 += 1

    i = 0
    j = ((n - silly_index) << 1) - 1
    while i < (silly_index << 1) and j > -1:
        total_sum = a_sums[i] + b_sums[j]
        if total_sum == desired_sum:
            return 1
        elif total_sum < desired_sum:
            i += 1
        else:
            j -= 1

    return 0


def schroeppel_shamir(p, desired_sum):
    """
    This function takes a list of numbers and a target number and returns a subset of the numbers that add up to the target number using Schroeppel-Shamir's algorithm.
    :param p:
    :param desired_sum:
    :return: list with the indexes of the numbers that add up to the target number
    """
    n = len(p)

    minHeap = []
    maxHeap = []

    L1Size = n // 4 + 1 if n % 4 != 0 else n // 4
    L2Size = n // 4 + (n % 4) // 2
    R2Size = n // 4 + (n % 4) // 3
    R1Size = n // 4

    L1 = p[:L1Size]
    L2 = p[L1Size:L1Size + L2Size]
    R2 = p[L1Size + L2Size:L1Size + L2Size + R2Size]
    R1 = p[L1Size + L2Size + R2Size:L1Size + L2Size + R2Size + R1Size]

    fancyL1Size = 1 << L1Size
    fancyL2Size = 1 << L2Size
    fancyR2Size = 1 << R2Size
    fancyR1Size = 1 << R1Size

    fancyL1 = merge_sort_subsets(L1)
    fancyL2 = merge_sort_subsets(L2)
    fancyR2 = merge_sort_subsets(R2)
    fancyR1 = merge_sort_subsets(R1)

    for k in range(fancyL1Size):
        heapq.heappush(minHeap, (fancyL1[k], (k, 0)))

    for k in range(fancyR2Size):
        heapq.heappush(maxHeap, (-(fancyR2[k] + fancyR1[fancyR1Size - 1]), (k, fancyR1Size - 1)))

    K = fancyL1Size * fancyL2Size + fancyR2Size * fancyR1Size

    for count in range(K):

        minTop, (iMin, jMin) = minHeap[0]
        maxTop, (iMax, jMax) = maxHeap[0]

        sum_val = minTop - maxTop

        if sum_val == desired_sum:
            return 1

        elif sum_val < desired_sum:
            heapq.heappop(minHeap)

            if jMin + 1 < fancyL2Size:
                heapq.heappush(minHeap, (fancyL1[iMin] + fancyL2[jMin + 1], (iMin, jMin + 1)))

        else:
            heapq.heappop(maxHeap)

            if jMax > 0:
                heapq.heappush(maxHeap, (-(fancyR2[iMax] + fancyR1[jMax - 1]), (iMax, jMax - 1)))

    return 0


def main():
    p = [1, 2, 3, 4]
    result = merge_sort_subsets(p)
    print(result)

    print()

    numbers = [3, 34, 4, 5, 12, 2]
    target = 9
    result = subset_sum_dp(numbers, target)
    print(result)

    print()

    numbers = [3, 34, 4, 5, 12, 2]
    target = 9
    result = horowitz_sahni(numbers, target)
    print(result)

    numbers = [3, 34, 4, 5, 12, 2]
    target = 9
    result = schroeppel_shamir(numbers, target)
    print(result)


if __name__ == "__main__":
    main()
