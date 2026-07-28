/* Strategy: Full WAL with CRC32 + fsync WAL + fsync data + rename + dir fsync + recovery */
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#include <arpa/inet.h>

static uint32_t crc_tab[256];

static void init_crc(void) {
    for (int i = 0; i < 256; i++) {
        uint32_t c = (uint32_t)i;
        for (int j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc_tab[i] = c;
    }
}

static uint32_t calc_crc(const uint8_t *d, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++)
        c = crc_tab[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static void do_recover(const char *dir) {
    char wal[4096], path[4096], tmp[4096];
    snprintf(wal, sizeof(wal), "%s/wal.log", dir);
    snprintf(path, sizeof(path), "%s/data.txt", dir);
    snprintf(tmp, sizeof(tmp), "%s/.data.txt.tmp", dir);

    int wfd = open(wal, O_RDONLY);
    if (wfd < 0) return;

    uint32_t nl, nc;
    if (read(wfd, &nl, 4) != 4 || read(wfd, &nc, 4) != 4) {
        close(wfd);
        unlink(wal);
        return;
    }
    uint32_t dlen = ntohl(nl);
    uint32_t stored_crc = ntohl(nc);
    if (dlen > 1048576) { close(wfd); unlink(wal); return; }

    uint8_t *buf = malloc(dlen);
    if (!buf) { close(wfd); return; }
    if ((size_t)read(wfd, buf, dlen) != dlen) {
        free(buf); close(wfd); unlink(wal); return;
    }
    close(wfd);

    if (calc_crc(buf, dlen) != stored_crc) {
        free(buf); unlink(wal); return;
    }

    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        write(fd, buf, dlen);
        fsync(fd);
        close(fd);
        rename(tmp, path);
        int dfd = open(dir, O_RDONLY);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
    }
    unlink(wal);
    free(buf);
}

static void do_write(const char *dir, const char *data) {
    char wal[4096], path[4096], tmp[4096];
    snprintf(wal, sizeof(wal), "%s/wal.log", dir);
    snprintf(path, sizeof(path), "%s/data.txt", dir);
    snprintf(tmp, sizeof(tmp), "%s/.data.txt.tmp", dir);

    size_t dlen = strlen(data);

    /* 1. Write WAL entry: 4-byte len + 4-byte CRC + payload */
    int wfd = open(wal, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) return;
    uint32_t nl = htonl((uint32_t)dlen);
    uint32_t nc = htonl(calc_crc((const uint8_t *)data, dlen));
    write(wfd, &nl, 4);
    write(wfd, &nc, 4);
    write(wfd, data, dlen);
    fsync(wfd);
    close(wfd);

    /* 2. Write temp data file */
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    write(fd, data, dlen);
    fsync(fd);
    close(fd);

    /* 3. Rename temp to final */
    rename(tmp, path);

    /* 4. Fsync directory */
    int dfd = open(dir, O_RDONLY);
    if (dfd >= 0) { fsync(dfd); close(dfd); }

    /* 5. Clear WAL */
    unlink(wal);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dir> [<data>|--recover]\n", argv[0]);
        return 1;
    }
    init_crc();
    mkdir(argv[1], 0755);

    if (argc == 3 && strcmp(argv[2], "--recover") == 0) {
        do_recover(argv[1]);
    } else if (argc == 3) {
        do_recover(argv[1]);
        do_write(argv[1], argv[2]);
    } else {
        fprintf(stderr, "Usage: %s <dir> [<data>|--recover]\n", argv[0]);
        return 1;
    }
    return 0;
}
