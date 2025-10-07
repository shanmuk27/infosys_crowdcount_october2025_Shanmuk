def divi(n1, n2):
    return "Zero division error" if n2 == 0 else n1 / n2

def calculator(num1, num2, op):
    match op:
        case "+":
            return num1 + num2
        case "-":
            return num1 - num2
        case "*":
            return num1 * num2
        case "/":
            return divi(num1, num2)
        case "%":
            return num1 % num2
        case "**":
            return num1 ** num2
        case _:
            return "Invalid operator"

n1 = float(input("Enter first number: "))
n2 = float(input("Enter second number: "))
op = input("Enter operation (+, -, *, /, %, **): ").strip()
res = calculator(n1, n2, op)
print(f"{n1} {op} {n2} = {res}")
