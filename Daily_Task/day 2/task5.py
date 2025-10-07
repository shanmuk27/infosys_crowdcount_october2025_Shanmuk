A = {1, 2, 3}
B = {3, 4, 5}
print(A | B)        # {1, 2, 3, 4, 5}
print(A.union(B))   # {1, 2, 3, 4, 5}
print(A & B)          # {3}
print(A.intersection(B))  # {3}
print(A - B)          # {1, 2}
print(B - A)          # {4, 5}
print(A ^ B)              # {1, 2, 4, 5}
print(A.symmetric_difference(B))  # {1, 2, 4, 5}

A = {1, 2}
B = {1, 2, 3}

print(A.issubset(B))   # True
print(B.issuperset(A)) # True
