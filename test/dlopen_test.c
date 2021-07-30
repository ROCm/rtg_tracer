#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    char *lib = getenv("HSA_TOOLS_LIB");
    if (lib) {
        void *ret = dlopen(lib, RTLD_LAZY);
        if (ret == NULL) {
            printf("dlopen failed: %s\n", dlerror());
            return EXIT_FAILURE;
        }
    }
    else {
        printf("getenv HSA_TOOLS_LIB failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
