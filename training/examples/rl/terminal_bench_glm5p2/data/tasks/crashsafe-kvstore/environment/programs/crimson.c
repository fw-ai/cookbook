/* Strategy: Temp file + fsync data + rename, but no directory fsync */
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

    char path[4096], tmp[4096];
    snprintf(path, sizeof(path), "%s/data.txt", argv[1]);
    snprintf(tmp, sizeof(tmp), "%s/.data.txt.tmp", argv[1]);

    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    write(fd, argv[2], strlen(argv[2]));
    fsync(fd);
    close(fd);
    rename(tmp, path);
    return 0;
}
