s={"hello",20,"e",2}
s.add("a")
s.update((8,"b"))
s2=s.copy()
s.clear()
s2.discard("a")
print(s)
print(s2)