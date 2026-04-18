#include <stdio.h>

int example(int a, int b) {
    for(int i = 0; i < 1000; i++) {
        a += i;
    }

    for (int i = 0; i<5; i++) {
        b ^= i;
    }
    return a + b;
}

int main() {
    int a = 5;
    int b = 10;
    int result = example(a, b);
    printf("Result: %d\n", result);
    return 0;
}