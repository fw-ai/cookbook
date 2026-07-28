#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Imported from libvault.so */
extern void gateway(void);

void pwnme(char *pivot_buf) {
    char buf[40];
    memset(buf, 0, sizeof(buf));

    printf("Pivot location: %p\n", pivot_buf);

    printf("Stage 2:\n> ");
    read(STDIN_FILENO, pivot_buf, 256);

    printf("Overflow:\n> ");
    read(STDIN_FILENO, buf, 96);
}

/* Ensures gateway has a PLT entry without being called in normal flow */
void uselessFunction(void) {
    gateway();
    exit(1);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    void *heap = malloc(0x100000);
    char *pivot = (char *)heap + 0xfff00;

    printf("=== Vault Challenge ===\n");
    pwnme(pivot);

    free(heap);
    return 0;
}
