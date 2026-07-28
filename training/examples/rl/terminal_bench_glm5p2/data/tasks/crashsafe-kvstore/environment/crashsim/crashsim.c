/*
 * LD_PRELOAD crash injection library.
 *
 * Intercepts write(), fsync(), rename(), and unlink() calls.
 * Terminates the process (simulating a crash) at a configurable point.
 *
 * Environment variables:
 *   CRASHSIM_SYSCALL  — which call to crash on: "write", "fsync", "rename", "unlink"
 *   CRASHSIM_NTH      — 1-based invocation count at which to crash (exit before the call)
 *
 * For write(), only calls with fd > 2 are counted (skips stdout/stderr).
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ssize_t (*real_write)(int, const void *, size_t) = NULL;
static int (*real_fsync)(int) = NULL;
static int (*real_rename)(const char *, const char *) = NULL;
static int (*real_unlink)(const char *) = NULL;

static const char *target = NULL;
static int crash_nth = 0;
static int cnt_write = 0;
static int cnt_fsync = 0;
static int cnt_rename = 0;
static int cnt_unlink = 0;
static int ready = 0;

__attribute__((constructor))
static void crashsim_init(void) {
    real_write  = dlsym(RTLD_NEXT, "write");
    real_fsync  = dlsym(RTLD_NEXT, "fsync");
    real_rename = dlsym(RTLD_NEXT, "rename");
    real_unlink = dlsym(RTLD_NEXT, "unlink");
    target = getenv("CRASHSIM_SYSCALL");
    const char *n = getenv("CRASHSIM_NTH");
    if (n) crash_nth = atoi(n);
    ready = 1;
}

ssize_t write(int fd, const void *buf, size_t count) {
    if (!ready) crashsim_init();
    if (fd > 2 && target && strcmp(target, "write") == 0) {
        cnt_write++;
        if (cnt_write == crash_nth)
            _exit(137);
    }
    return real_write(fd, buf, count);
}

int fsync(int fd) {
    if (!ready) crashsim_init();
    if (target && strcmp(target, "fsync") == 0) {
        cnt_fsync++;
        if (cnt_fsync == crash_nth)
            _exit(137);
    }
    return real_fsync(fd);
}

int rename(const char *oldpath, const char *newpath) {
    if (!ready) crashsim_init();
    if (target && strcmp(target, "rename") == 0) {
        cnt_rename++;
        if (cnt_rename == crash_nth)
            _exit(137);
    }
    return real_rename(oldpath, newpath);
}

int unlink(const char *path) {
    if (!ready) crashsim_init();
    if (target && strcmp(target, "unlink") == 0) {
        cnt_unlink++;
        if (cnt_unlink == crash_nth)
            _exit(137);
    }
    return real_unlink(path);
}
