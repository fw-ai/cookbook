#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void gateway(void) {
    printf("Gateway checkpoint reached.\n");
}

void vault(uint64_t key1, uint64_t key2, uint64_t key3) {
    if (key1 == 0xdeadbeefcafebabeULL &&
        key2 == 0x1337c0de1337c0deULL &&
        key3 == 0xfeedface0badf00dULL) {
        FILE *f = fopen("/app/flag.txt", "r");
        if (f) {
            char buf[256];
            memset(buf, 0, sizeof(buf));
            if (fgets(buf, sizeof(buf), f)) {
                size_t len = strlen(buf);
                if (len > 0 && buf[len - 1] == '\n')
                    buf[len - 1] = '\0';
                printf("%s\n", buf);
            }
            fclose(f);
        } else {
            printf("Error: cannot open flag file\n");
        }
    } else {
        printf("Access denied: incorrect keys\n");
        printf("Received: 0x%lx 0x%lx 0x%lx\n", key1, key2, key3);
    }
    exit(0);
}
