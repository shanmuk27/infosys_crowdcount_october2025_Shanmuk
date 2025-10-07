d = {"a": 1, "b": 2, "c": 3}

print(d.keys())
print(d.values())
print(d.items())

print(d.get("b"))
print(d.get("z", "Not Found"))

d.update({"d": 4})
print(d)

d.pop("a")
print(d)

d.clear()
print(d)
