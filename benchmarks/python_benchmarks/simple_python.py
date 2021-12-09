import random


def main():
    j = 0
    for i in range(1, 40000):
        j += random.randint(0, 400000) % i
    print(j)


if __name__ == '__main__':
    main()
