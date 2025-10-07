def add_numbers(*args):
    result = 0
    for num in args:
        result += num
    return result

def show_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

print(add_numbers(*[1,2,3,4]))
show_info(**{"name": "Alice", "age": 20, "grade": "A"})
