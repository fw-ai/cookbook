/* Strategy: Direct overwrite - no atomicity, no durability */
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <dir> <data>\n", argv[0]);
        return 1;
    }
    mkdir(argv[1], 0755);

    char path[4096];
    snprintf(path, sizeof(path), "%s/data.txt", argv[1]);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    write(fd, argv[2], strlen(argv[2]));
    close(fd);
    return 0;
}
