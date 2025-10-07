import os
import math
import random

# Random module
print(random.random())
print(random.randint(1,100))
print(random.randrange(5))
lis=[2,1,3,4]
print(random.choice(lis))
random.shuffle(lis)
print(lis)

# Math module
print(math.pi)
print(math.e)
print(math.degrees(45))
print(math.sin(45))
print(math.sqrt(4))
print(math.log(10,10))
print(math.exp(2))
print(math.pow(2,3))
print(math.radians(3))
print(math.ceil(4.56))
print(math.floor(4.24))

# OS module
folder_path = r'C:\Users\pooji\OneDrive\Desktop\sai\inf\example'

# Create folder only if it does not exist
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
    print("Folder created")
else:
    print("Folder already exists")

# Change directory
os.chdir(folder_path)
print("Current directory:", os.getcwd())

# Remove folder (go back first to avoid error)
os.chdir(r'C:\Users\pooji\OneDrive\Desktop\sai\inf')
if os.path.exists(folder_path):
    os.rmdir(folder_path)
    print("Folder removed")
