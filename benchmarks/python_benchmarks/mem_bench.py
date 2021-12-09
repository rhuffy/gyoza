import random


def main():
    j = 1
    big_array = [i for i in range(1000000)]
    for i in big_array:
        random_elt = random.choice(big_array)
        j = (random_elt + i) % j
    print(j)


if __name__ == '__main__':
    main()
