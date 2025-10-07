my_set = {10, 20, 30, 40}

print("Original set:", my_set)

my_set.add(50)
print("After add(50):", my_set)

my_set.update([60, 70], {80, 90})
print("After update:", my_set)

my_set.remove(30)
print("After remove(30):", my_set)

my_set.discard(100)
print("After discard(100):", my_set)

removed = my_set.pop()
print("Popped element:", removed)
print("After pop():", my_set)

temp = my_set.copy()
temp.clear()
print("After clear():", temp)

new_set = my_set.copy()
print("Copy of set:", new_set)

A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A |= B
print("Union Update:", A)

A = {1, 2, 3, 4}
A.intersection_update(B)
print("Intersection Update:", A)

A = {1, 2, 3, 4}
A.difference_update(B)
print("Difference Update:", A)

A = {1, 2, 3, 4}
A.symmetric_difference_update(B)
print("Symmetric Difference Update:", A)