/*
 * Vinaigrette Signature Verifier
 * Reads the VNRT binary public key and hash target, verifies a candidate signature.
 * Usage: ./verifier <signature_file>
 * Exit 0 = valid, Exit 1 = invalid, Exit 2 = error
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_N 16
#define MAX_M 8
#define MAX_K 4

typedef struct {
    uint8_t q, n, m, o, k;
    int P[MAX_M][MAX_N][MAX_N];
    int hash_target[MAX_M];
} Challenge;

static int read_binary_key(const char *path, Challenge *ch) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VNRT", 4) != 0) {
        fprintf(stderr, "Invalid magic in %s\n", path);
        fclose(f); return -2;
    }

    uint8_t version;
    fread(&version, 1, 1, f);
    fread(&ch->q, 1, 1, f);
    fread(&ch->n, 1, 1, f);
    fread(&ch->m, 1, 1, f);
    fread(&ch->o, 1, 1, f);
    fread(&ch->k, 1, 1, f);

    uint8_t reserved[2];
    fread(reserved, 1, 2, f);

    memset(ch->P, 0, sizeof(ch->P));
    for (int mat = 0; mat < ch->m; mat++) {
        for (int i = 0; i < ch->n; i++) {
            for (int j = i; j < ch->n; j++) {
                uint8_t val;
                if (fread(&val, 1, 1, f) != 1) {
                    fprintf(stderr, "Truncated key data\n");
                    fclose(f); return -3;
                }
                ch->P[mat][i][j] = val;
            }
        }
    }
    fclose(f);
    return 0;
}

static int read_hash_target(const char *path, Challenge *ch) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    for (int i = 0; i < ch->m; i++) {
        uint8_t val;
        if (fread(&val, 1, 1, f) != 1) {
            fprintf(stderr, "Truncated hash target\n");
            fclose(f); return -2;
        }
        ch->hash_target[i] = val;
    }
    fclose(f);
    return 0;
}

static int eval_quadratic(int P[MAX_N][MAX_N], const int *x, int q, int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            s += (long long)P[i][j] * x[i] * x[j];
        }
    }
    return (int)(((s % q) + q) % q);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <signature_file>\n", argv[0]);
        return 2;
    }

    Challenge ch;
    if (read_binary_key("/challenge/public_key.bin", &ch) != 0) return 2;
    if (read_hash_target("/challenge/hash_target.bin", &ch) != 0) return 2;

    FILE *sf = fopen(argv[1], "r");
    if (!sf) {
        fprintf(stderr, "Cannot open signature file: %s\n", argv[1]);
        return 2;
    }

    int total = ch.k * ch.n;
    int *sig = calloc(total, sizeof(int));
    for (int i = 0; i < total; i++) {
        if (fscanf(sf, "%d", &sig[i]) != 1) {
            fprintf(stderr, "Signature too short: expected %d values, got %d\n", total, i);
            free(sig); fclose(sf); return 2;
        }
        if (sig[i] < 0 || sig[i] >= ch.q) {
            fprintf(stderr, "INVALID: value %d at position %d out of range [0, %d)\n",
                    sig[i], i, ch.q);
            free(sig); fclose(sf); return 1;
        }
    }
    fclose(sf);

    int computed[MAX_M];
    memset(computed, 0, sizeof(computed));
    for (int i = 0; i < ch.k; i++) {
        int *s_i = &sig[i * ch.n];
        for (int ki = 0; ki < ch.m; ki++) {
            computed[ki] = (computed[ki] + eval_quadratic(ch.P[ki], s_i, ch.q, ch.n)) % ch.q;
        }
    }

    int valid = 1;
    for (int ki = 0; ki < ch.m; ki++) {
        if (computed[ki] != ch.hash_target[ki]) { valid = 0; break; }
    }

    if (valid) {
        printf("VALID: Signature verifies correctly.\n");
        free(sig);
        return 0;
    } else {
        printf("INVALID: Verification equation P*(s) != H(msg) mod %d\n", ch.q);
        printf("  Computed:");
        for (int ki = 0; ki < ch.m; ki++) printf(" %d", computed[ki]);
        printf("\n  Expected:");
        for (int ki = 0; ki < ch.m; ki++) printf(" %d", ch.hash_target[ki]);
        printf("\n");
        free(sig);
        return 1;
    }
}
