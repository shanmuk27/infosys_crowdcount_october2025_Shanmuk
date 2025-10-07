class Calculator:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2

    def add(self):
        print("Addition:", self.n1 + self.n2)

    def subtract(self):
        print("Subtraction:", self.n1 - self.n2)

    def multiply(self):
        print("Multiplication:", self.n1 * self.n2)

    def divide(self):
        if self.n2 == 0:
            print("Division by zero!!!")
        else:
            print("Division:", self.n1 / self.n2)

calc = Calculator(12, 4)
calc.add()
calc.subtract()
calc.multiply()
calc.divide()

