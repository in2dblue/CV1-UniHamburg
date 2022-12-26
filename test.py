def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# print(quicksort([3,6,8,10,1,2,1]))
# a = [3,6,8,10,1,2,1]
# pivot = a[len(a) // 2]
# print(pivot)
# left = [x for x in a if x > pivot]
# print(left)

print(7//2)