# Python script to generate a file with numbers from 0 to n-1, each on a new line


n = int(input("请输入一个整数n："))
with open("list.txt", "w") as file:
    for i in range(n):
        file.write(f"{i}\n")
print("已将0到n-1的整数每行一个顺序存到list.txt文件中。")

