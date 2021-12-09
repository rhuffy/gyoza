#include <stdio.h>

int main()
{
    int j = 1;
    for (int i = 0; i < 10000; i++)
    {
        if (i < 1000)
        {
            j += i;
        }
        else
        {
            j = ((j % i) + i) % i;
        }
    }
    printf("j: %d", j);
    return 0;
}