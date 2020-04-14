#! /usr/bin/env python
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import argparse
sys.path.append('maint/')
import yutils

indentation = 0
num_paren_open = 0

def display(*argv):
    for x in range(indentation):
        OUTFILE.write("    ")
    for arg in argv:
        OUTFILE.write(arg)


########################################################################################
##### Type-specific functions
########################################################################################

## hvector routines
def hvector(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    display("intptr_t stride%d = %s->u.hvector.stride / sizeof(%s);\n" % (suffix, dtp, b))
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    if (last != 1):
        s += " + x%d * stride%d + x%d * extent%d" % (idx, suffix, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + x%d * stride%d + x%d" % (idx, suffix, idx + 1)
        need_extent = False
    idx = idx + 2

## blkhindx routines
def blkhindx(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    display("intptr_t *array_of_displs%d = %s->u.blkhindx.array_of_displs;\n" % (suffix, dtp))
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    if (last != 1):
        s += " + array_of_displs%d[x%d] / sizeof(%s) + x%d * extent%d" % \
             (suffix, idx, b, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + array_of_displs%d[x%d] / sizeof(%s) + x%d" % (suffix, idx, b, idx + 1)
        need_extent = False
    idx = idx + 2

## hindexed routines
def hindexed(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    display("intptr_t *array_of_displs%d = %s->u.hindexed.array_of_displs;\n" % (suffix, dtp))
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    if (last != 1):
        s += " + array_of_displs%d[x%d] / sizeof(%s) + x%d * extent%d" % \
             (suffix, idx, b, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + array_of_displs%d[x%d] / sizeof(%s) + x%d" % (suffix, idx, b, idx + 1)
        need_extent = False
    idx = idx + 2

## dup routines
def dup(suffix, dtp, b, last):
    global need_extent
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    need_extent = False

## contig routines
def contig(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    display("intptr_t stride%d = %s->u.contig.child->extent / sizeof(%s);\n" % (suffix, dtp, b))
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    need_extent = False
    s += " + x%d * stride%d" % (idx, suffix)
    idx = idx + 1

# resized routines
def resized(suffix, dtp, b, last):
    global need_extent
    if (need_extent == True):
        display("uintptr_t extent%d = %s->extent / sizeof(%s);\n" % (suffix, dtp, b))
    need_extent = False


## loop through the derived and basic types to generate individual
## pack functions
builtin_types = [ "char", "wchar_t", "int", "short", "long", "long long", "int8_t", "int16_t", \
                  "int32_t", "int64_t", "float", "double" ]
derived_types = [ "hvector", "blkhindx", "hindexed", "dup", "contig", "resized" ]
derived_maps = {
    "hvector": hvector,
    "blkhindx": blkhindx,
    "hindexed": hindexed,
    "dup": dup,
    "contig": contig,
    "resized": resized,
}

builtin_maps = {
    "YAKSA_TYPE__UNSIGNED_CHAR": "char",
    "YAKSA_TYPE__UNSIGNED": "int",
    "YAKSA_TYPE__UNSIGNED_SHORT": "short",
    "YAKSA_TYPE__UNSIGNED_LONG": "long",
    "YAKSA_TYPE__LONG_DOUBLE": "double",
    "YAKSA_TYPE__UNSIGNED_LONG_LONG": "long_long",
    "YAKSA_TYPE__UINT8_T": "int8_t",
    "YAKSA_TYPE__UINT16_T": "int16_t",
    "YAKSA_TYPE__UINT32_T": "int32_t",
    "YAKSA_TYPE__UINT64_T": "int64_t",
    "YAKSA_TYPE__C_COMPLEX": "float",
    "YAKSA_TYPE__C_DOUBLE_COMPLEX": "double",
    "YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX": "double",
    "YAKSA_TYPE__BYTE": "int8_t"
}


########################################################################################
##### Core kernels
########################################################################################
def generate_kernels(b, darray):
    global indentation
    global need_extent
    global s
    global idx

    for func in "pack","unpack":
        ##### figure out the function name to use
        funcprefix = "%s_" % func
        for d in darray:
            funcprefix = funcprefix + "%s_" % d
        funcprefix = funcprefix + b.replace(" ", "_")

        ##### generate the CUDA kernel
        if (len(darray)):
            display("__global__ void yaksuri_cudai_kernel_%s(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)\n" % funcprefix)
            display("{\n")
            indentation += 1
            display("const %s *__restrict__ sbuf = (const %s *) inbuf;\n" % (b, b));
            display("%s *__restrict__ dbuf = (%s *) outbuf;\n" % (b, b));
            display("uintptr_t extent = md->extent / sizeof(%s);\n" % b)
            display("uintptr_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n")
            display("uintptr_t res = idx;\n")
            display("uintptr_t inner_elements = md->num_elements;\n")
            OUTFILE.write("\n")

            display("if (idx >= (count * inner_elements))\n")
            display("    return;\n")
            OUTFILE.write("\n")

            # copy loop
            idx = 0
            md = "md"
            for d in darray:
                if (d == "hvector" or d == "blkhindx" or d == "hindexed" or \
                    d == "contig"):
                    display("uintptr_t x%d = res / inner_elements;\n" % idx)
                    idx = idx + 1
                    display("res %= inner_elements;\n")
                    display("inner_elements /= %s->u.%s.count;\n" % (md, d))
                    OUTFILE.write("\n")

                if (d == "hvector" or d == "blkhindx"):
                    display("uintptr_t x%d = res / inner_elements;\n" % idx)
                    idx = idx + 1
                    display("res %= inner_elements;\n")
                    display("inner_elements /= %s->u.%s.blocklength;\n" % (md, d))
                elif (d == "hindexed"):
                    display("uintptr_t x%d;\n" % idx)
                    display("for (int i = 0; i < %s->u.%s.count; i++) {\n" % (md, d))
                    display("    uintptr_t in_elems = %s->u.%s.array_of_blocklengths[i] *\n" % (md, d))
                    display("                         %s->u.%s.child->num_elements;\n" % (md, d))
                    display("    if (res < in_elems) {\n")
                    display("        x%d = i;\n" % idx)
                    display("        res %= in_elems;\n")
                    display("        inner_elements = %s->u.%s.child->num_elements;\n" % (md, d))
                    display("        break;\n")
                    display("    } else {\n")
                    display("        res -= in_elems;\n")
                    display("    }\n")
                    display("}\n")
                    idx = idx + 1
                    OUTFILE.write("\n")

                md = "%s->u.%s.child" % (md, d)

            display("uintptr_t x%d = res;\n" % idx)
            OUTFILE.write("\n")

            dtp = "md"
            s = "x0 * extent"
            idx = 1
            x = 1
            need_extent = False
            for d in darray:
                if (x == len(darray)):
                    last = 1
                else:
                    last = 0
                derived_maps[d](x, dtp, b, last)
                x = x + 1
                dtp = dtp + "->u.%s.child" % d

            if (func == "pack"):
                display("dbuf[idx] = sbuf[%s];\n" % s)
            else:
                display("dbuf[%s] = sbuf[idx];\n" % s)

            indentation -= 1
            display("}\n\n")

            # generate the host function
            OUTFILE.write("void yaksuri_cudai_%s(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)\n" % funcprefix)
            OUTFILE.write("{\n")
            OUTFILE.write("    void *args[] = { &inbuf, &outbuf, &count, &md };\n")
            OUTFILE.write("    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_%s,\n" % funcprefix)
            OUTFILE.write("                dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);\n")
            OUTFILE.write("    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);\n")
            OUTFILE.write("}\n\n")


########################################################################################
##### Switch statement generation for pup function selection
########################################################################################
def child_type_str(typelist):
    s = "type"
    for x in typelist:
        s = s + "->u.%s.child" % x
    return s

def switcher_builtin_element(typelist, pupstr, key, val):
    global indentation
    display("case %s:\n" % key.upper())
    indentation += 1

    if (len(typelist) == 0):
        t = ""
    else:
        t = typelist.pop()

    if (t == ""):
        nesting_level = 0
    else:
        nesting_level = len(typelist) + 1

    display("if (max_nesting_level >= %d) {\n" % nesting_level)
    display("    cuda->pack = yaksuri_cudai_%s_%s;\n" % (pupstr, val))
    display("    cuda->unpack = yaksuri_cudai_un%s_%s;\n" % (pupstr, val))
    display("}\n")

    if (t != ""):
        typelist.append(t)
    display("break;\n")
    indentation -= 1

def switcher_builtin(typelist, pupstr):
    global indentation
    display("switch (%s->id) {\n" % child_type_str(typelist))
    indentation += 1

    for b in builtin_types:
        switcher_builtin_element(typelist, pupstr, "YAKSA_TYPE__%s" % b.replace(" ", "_"), b.replace(" ", "_"))
    for key in builtin_maps:
        switcher_builtin_element(typelist, pupstr, key, builtin_maps[key])

    display("default:\n")
    display("    break;\n")
    indentation -= 1
    display("}\n")

def switcher(typelist, pupstr, nests):
    global indentation

    display("switch (%s->kind) {\n" % child_type_str(typelist))

    for d in derived_types:
        indentation += 1
        if (nests > 1):
            display("case YAKSI_TYPE_KIND__%s:\n" % d.upper())
            indentation += 1
            typelist.append(d)
            switcher(typelist, pupstr + "_%s" % d, nests - 1)
            typelist.pop()
            display("break;\n")
            indentation -= 1
        indentation -= 1

    if (len(typelist)):
        indentation += 1
        display("case YAKSI_TYPE_KIND__BUILTIN:\n")
        indentation += 1
        switcher_builtin(typelist, pupstr)
        display("break;\n")
        indentation -= 2

    indentation += 1
    display("default:\n")
    display("    break;\n")
    indentation -= 1
    display("}\n")


########################################################################################
##### main function
########################################################################################
if __name__ == '__main__':
    ##### parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pup-max-nesting', type=int, default=3, help='maximum nesting levels to generate')
    args = parser.parse_args()
    if (args.pup_max_nesting < 0):
        parser.print_help()
        print
        print("===> ERROR: pup-max-nesting must be positive")
        sys.exit(1)

    ##### generate the list of derived datatype arrays
    darraylist = [ ]
    yutils.generate_darrays(derived_types, darraylist, args.pup_max_nesting)

    ##### generate the core pack/unpack kernels
    for b in builtin_types:
        filename = "src/backend/cuda/pup/yaksuri_cudai_pup_%s.cu" % b.replace(" ","_")
        yutils.copyright_c(filename)
        OUTFILE = open(filename, "a")
        OUTFILE.write("#include <string.h>\n")
        OUTFILE.write("#include <stdint.h>\n")
        OUTFILE.write("#include <wchar.h>\n")
        OUTFILE.write("#include <assert.h>\n")
        OUTFILE.write("#include <cuda.h>\n")
        OUTFILE.write("#include <cuda_runtime.h>\n")
        OUTFILE.write("#include \"yaksuri_cudai.h\"\n")
        OUTFILE.write("#include \"yaksuri_cudai_populate_pupfns.h\"\n")
        OUTFILE.write("\n")

        for darray in darraylist:
            generate_kernels(b, darray)

        OUTFILE.close()

    ##### generate the switching logic to select pup functions
    filename = "src/backend/cuda/pup/yaksuri_cudai_populate_pupfns.c"
    yutils.copyright_c(filename)
    OUTFILE = open(filename, "a")
    OUTFILE.write("#include <stdio.h>\n")
    OUTFILE.write("#include <stdlib.h>\n")
    OUTFILE.write("#include <wchar.h>\n")
    OUTFILE.write("#include \"yaksi.h\"\n")
    OUTFILE.write("#include \"yaksu.h\"\n")
    OUTFILE.write("#include \"yaksuri_cudai.h\"\n")
    OUTFILE.write("#include \"yaksuri_cudai_populate_pupfns.h\"\n")
    OUTFILE.write("\n")
    OUTFILE.write("int yaksuri_cudai_populate_pupfns(yaksi_type_s * type)\n")
    OUTFILE.write("{\n")
    OUTFILE.write("    int rc = YAKSA_SUCCESS;\n")
    OUTFILE.write("    yaksuri_cudai_type_s *cuda = (yaksuri_cudai_type_s *) type->backend.cuda.priv;\n")
    OUTFILE.write("\n")
    OUTFILE.write("    cuda->pack = NULL;\n")
    OUTFILE.write("    cuda->unpack = NULL;\n")
    OUTFILE.write("\n")
    OUTFILE.write("    char *str = getenv(\"YAKSA_ENV_MAX_NESTING_LEVEL\");\n")
    OUTFILE.write("    int max_nesting_level;\n")
    OUTFILE.write("    if (str) {\n")
    OUTFILE.write("        max_nesting_level = atoi(str);\n")
    OUTFILE.write("    } else {\n")
    OUTFILE.write("        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;\n")
    OUTFILE.write("    }\n")
    OUTFILE.write("\n")

    indentation += 1
    pupstr = "pack"
    typelist = [ ]
    switcher(typelist, pupstr, args.pup_max_nesting + 1)
    OUTFILE.write("\n")
    display("return rc;\n")
    indentation -= 1
    display("}\n")

    OUTFILE.close()

    ##### generate the header file declarations
    filename = "src/backend/cuda/pup/yaksuri_cudai_populate_pupfns.h"
    yutils.copyright_c(filename)
    OUTFILE = open(filename, "a")
    OUTFILE.write("#ifndef YAKSURI_CUDAI_POPULATE_PUPFNS_H_INCLUDED\n")
    OUTFILE.write("#define YAKSURI_CUDAI_POPULATE_PUPFNS_H_INCLUDED\n")
    OUTFILE.write("\n")
    OUTFILE.write("#include <string.h>\n")
    OUTFILE.write("#include <stdint.h>\n")
    OUTFILE.write("#include \"yaksi.h\"\n")
    OUTFILE.write("#include \"yaksuri_cudai.h\"\n")
    OUTFILE.write("\n")
    OUTFILE.write("#ifdef __cplusplus\n")
    OUTFILE.write("extern \"C\"\n")
    OUTFILE.write("{\n")
    OUTFILE.write("#endif\n")
    OUTFILE.write("\n")

    for b in builtin_types:
        for darray in darraylist:
            for func in "pack","unpack":
                ##### figure out the function name to use
                s = "void yaksuri_cudai_%s_" % func
                for d in darray:
                    s = s + "%s_" % d
                s = s + b.replace(" ", "_")
                OUTFILE.write("%s" % s)
                OUTFILE.write("(const void *inbuf, ")
                OUTFILE.write("void *outbuf, ")
                OUTFILE.write("uintptr_t count, ")
                OUTFILE.write("yaksuri_cudai_md_s *md, ")
                OUTFILE.write("int n_threads, ")
                OUTFILE.write("int n_blocks_x, ")
                OUTFILE.write("int n_blocks_y, ")
                OUTFILE.write("int n_blocks_z, ")
                OUTFILE.write("int device);\n")

    OUTFILE.write("\n")
    OUTFILE.write("#ifdef __cplusplus\n")
    OUTFILE.write("}\n")
    OUTFILE.write("#endif\n")
    OUTFILE.write("\n")
    OUTFILE.write("#endif  /* YAKSURI_CUDAI_POPULATE_PUPFNS_H_INCLUDED */\n")
    OUTFILE.close()
