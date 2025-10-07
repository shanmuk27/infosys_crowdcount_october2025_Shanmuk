import os

# 1. Open a new file to write to it. 'w' creates the file.
file = open("my_first_file.txt", "w")

# 2. Write some lines of text into the file.
file.write("Hello, this is my first line.\n")
file.write("This is the second line.\n")

# 3. Always close the file when you're done.
file.close()

# 4. Now, open the same file to add more text. 'a' is for append.
file = open("my_first_file.txt", "a")
file.write("This line was added later.\n")
file.close()

# 5. Open the file to read its content. 'r' is for read.
file = open("my_first_file.txt", "r")

# 6. Read everything in the file and print it to the screen.
content = file.read()
print("Here is the content of the file:")
print(content)

# 7. Close the file again.
file.close()

# 8. Finally, delete the file from the computer.
os.remove("my_first_file.txt")

print("\nFile has been deleted.")