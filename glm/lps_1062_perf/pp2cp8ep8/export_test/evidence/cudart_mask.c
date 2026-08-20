#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

/* cu12.8-image + cu13-venv mask.
 * - flashinfer.jit loads the SYSTEM libcudart.so.12 by ABSOLUTE path at import
 *   (ctypes, must succeed) -> redirect those to the venv cu13 runtime.
 * - the cudnn dsatopk1 shim PROBES dlopen("libcudart.so.12") BY SONAME and
 *   hard-fails if it also succeeds -> fail soname probes so only the cu13
 *   runtime is visible.
 * Result: process stays cu13-pure; flashinfer import survives.
 */
static void *(*real_dlopen)(const char *, int) = NULL;
static const char *CU13 = "/root/.cache/user_artifacts/trainers_main/server/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13";

void *dlopen(const char *filename, int flag) {
    if (!real_dlopen) real_dlopen = dlsym(RTLD_NEXT, "dlopen");
    if (filename && strstr(filename, "libcudart.so.12")) {
        if (strchr(filename, (int)47))  /* absolute/relative path: redirect */
            return real_dlopen(CU13, flag);
        return NULL;  /* soname probe: fail */
    }
    return real_dlopen(filename, flag);
}
