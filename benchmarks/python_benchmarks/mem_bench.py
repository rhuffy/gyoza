import random


def main():
    j = 1
    big_array = [i for i in range(50000)]
    big_array_2 = [i for i in range(50000)]
    for i in big_array:
        random_elt = random.choice(big_array)
        random_elt_2 = random.choice(big_array_2)
        random_elt = min(random_elt, random_elt_2)
        j = (random_elt + i) % j
    print(j)


if __name__ == '__main__':
    main()
