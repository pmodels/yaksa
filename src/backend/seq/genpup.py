#! /usr/bin/env python
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import argparse

sys.path.append('maint/')
import yutils

sys.path.append('src/backend/')
import gencomm


########################################################################################
##### Global variables
########################################################################################

num_paren_open = 0
builtin_types = [ "char", "wchar_t", "int", "short", "long", "long long", "int8_t", "int16_t", \
                  "int32_t", "int64_t", "float", "double", "long double" ]
blklens = [ "1", "2", "3", "4", "5", "6", "7", "8", "generic" ]
builtin_maps = {
    "YAKSA_TYPE__UNSIGNED_CHAR": "char",
    "YAKSA_TYPE__UNSIGNED": "int",
    "YAKSA_TYPE__UNSIGNED_SHORT": "short",
    "YAKSA_TYPE__UNSIGNED_LONG": "long",
    "YAKSA_TYPE__UNSIGNED_LONG_LONG": "long_long",
    "YAKSA_TYPE__UINT8_T": "int8_t",
    "YAKSA_TYPE__UINT16_T": "int16_t",
    "YAKSA_TYPE__UINT32_T": "int32_t",
    "YAKSA_TYPE__UINT64_T": "int64_t",
    "YAKSA_TYPE__BYTE": "int8_t"
}


########################################################################################
##### Type-specific functions
########################################################################################

## hvector routines
def hvector_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "int count%d = %s->u.hvector.count;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "int blocklength%d ATTRIBUTE((unused)) = %s->u.hvector.blocklength;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "intptr_t stride%d = %s->u.hvector.stride / sizeof(%s);\n" % (nesting, dtp, b))
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def hvector(suffix, b, blklen, last):
    global num_paren_open
    num_paren_open += 2
    yutils.display(OUTFILE, "for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    if (blklen == "generic"):
        yutils.display(OUTFILE, "for (int k%d = 0; k%d < blocklength%d; k%d++) {\n" % (suffix, suffix, suffix, suffix))
    else:
        yutils.display(OUTFILE, "for (int k%d = 0; k%d < %s; k%d++) {\n" % (suffix, suffix, blklen, suffix))
    global s
    if (last != 1):
        s += " + j%d * stride%d + k%d * extent%d" % (suffix, suffix, suffix, suffix + 1)
    else:
        s += " + j%d * stride%d + k%d" % (suffix, suffix, suffix)

## blkhindx routines
def blkhindx_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "int count%d = %s->u.blkhindx.count;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "int blocklength%d ATTRIBUTE((unused)) = %s->u.blkhindx.blocklength;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "intptr_t *restrict array_of_displs%d = %s->u.blkhindx.array_of_displs;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def blkhindx(suffix, b, blklen, last):
    global num_paren_open
    num_paren_open += 2
    yutils.display(OUTFILE, "for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    if (blklen == "generic"):
        yutils.display(OUTFILE, "for (int k%d = 0; k%d < blocklength%d; k%d++) {\n" % (suffix, suffix, suffix, suffix))
    else:
        yutils.display(OUTFILE, "for (int k%d = 0; k%d < %s; k%d++) {\n" % (suffix, suffix, blklen, suffix))
    global s
    if (last != 1):
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d * extent%d" % \
             (suffix, suffix, b, suffix, suffix + 1)
    else:
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d" % (suffix, suffix, b, suffix)

## hindexed routines
def hindexed_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "int count%d = %s->u.hindexed.count;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "int *restrict array_of_blocklengths%d = %s->u.hindexed.array_of_blocklengths;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "intptr_t *restrict array_of_displs%d = %s->u.hindexed.array_of_displs;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def hindexed(suffix, b, blklen, last):
    global num_paren_open
    num_paren_open += 2
    yutils.display(OUTFILE, "for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    yutils.display(OUTFILE, "for (int k%d = 0; k%d < array_of_blocklengths%d[j%d]; k%d++) {\n" % \
            (suffix, suffix, suffix, suffix, suffix))
    global s
    if (last != 1):
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d * extent%d" % \
             (suffix, suffix, b, suffix, suffix + 1)
    else:
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d" % (suffix, suffix, b, suffix)

## dup routines
def dup_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def dup(suffix, b, blklen, last):
    pass

## contig routines
def contig_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "int count%d = %s->u.contig.count;\n" % (nesting, dtp))
    yutils.display(OUTFILE, "intptr_t stride%d = %s->u.contig.child->extent / sizeof(%s);\n" % (nesting, dtp, b))
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def contig(suffix, b, blklen, last):
    global num_paren_open
    num_paren_open += 1
    yutils.display(OUTFILE, "for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    global s
    s += " + j%d * stride%d" % (suffix, suffix)

# resized routines
def resized_decl(nesting, dtp, b):
    yutils.display(OUTFILE, "uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def resized(suffix, b, blklen, last):
    pass


########################################################################################
##### Core kernels
########################################################################################
def generate_kernels(b, darray, blklen):
    global num_paren_open
    global s

    # we don't need pup kernels for basic types
    if (len(darray) == 0):
        return

    # individual blocklength optimization is only for
    # hvector and blkhindx
    if (darray[-1] != "hvector" and darray[-1] != "blkhindx" and blklen != "generic"):
        return

    for func in "pack","unpack":
        ##### figure out the function name to use
        s = "int yaksuri_seqi_%s_" % func
        for d in darray:
            s = s + "%s_" % d
        # hvector and hindexed get blklen-specific function names
        if (darray[-1] != "hvector" and darray[-1] != "blkhindx"):
            s = s + b.replace(" ", "_")
        else:
            s = s + "blklen_%s_" % blklen + b.replace(" ", "_")
        yutils.display(OUTFILE, "%s(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)\n" % s),
        yutils.display(OUTFILE, "{\n")


        ##### variable declarations
        # generic variables
        yutils.display(OUTFILE, "int rc = YAKSA_SUCCESS;\n");
        yutils.display(OUTFILE, "const %s *restrict sbuf = (const %s *) inbuf;\n" % (b, b));
        yutils.display(OUTFILE, "%s *restrict dbuf = (%s *) outbuf;\n" % (b, b));
        yutils.display(OUTFILE, "uintptr_t extent ATTRIBUTE((unused)) = type->extent / sizeof(%s);\n" % b)
        yutils.display(OUTFILE, "\n");

        # variables specific to each nesting level
        s = "type"
        for x in range(len(darray)):
            getattr(sys.modules[__name__], "%s_decl" % darray[x])(x + 1, s, b)
            yutils.display(OUTFILE, "\n")
            s = s + "->u.%s.child" % darray[x]


        ##### non-hvector and non-blkhindx
        yutils.display(OUTFILE, "uintptr_t idx = 0;\n")
        yutils.display(OUTFILE, "for (int i = 0; i < count; i++) {\n")
        num_paren_open += 1
        s = "i * extent"
        for x in range(len(darray)):
            if (x != len(darray) - 1):
                getattr(sys.modules[__name__], darray[x])(x + 1, b, "generic", 0)
            else:
                getattr(sys.modules[__name__], darray[x])(x + 1, b, blklen, 1)

        if (func == "pack"):
            yutils.display(OUTFILE, "dbuf[idx++] = sbuf[%s];\n" % s)
        else:
            yutils.display(OUTFILE, "dbuf[%s] = sbuf[idx++];\n" % s)
        for x in range(num_paren_open):
            yutils.display(OUTFILE, "}\n")
        num_paren_open = 0
        yutils.display(OUTFILE, "\n");
        yutils.display(OUTFILE, "return rc;\n")
        yutils.display(OUTFILE, "}\n\n")


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
    yutils.generate_darrays(gencomm.derived_types, darraylist, args.pup_max_nesting)

    ##### generate the core pack/unpack kernels
    for b in builtin_types:
        filename = "src/backend/seq/pup/yaksuri_seqi_pup_%s.c" % b.replace(" ","_")
        yutils.copyright_c(filename)
        OUTFILE = open(filename, "a")
        yutils.display(OUTFILE, "#include <string.h>\n")
        yutils.display(OUTFILE, "#include <stdint.h>\n")
        yutils.display(OUTFILE, "#include <wchar.h>\n")
        yutils.display(OUTFILE, "#include \"yaksuri_seqi_pup.h\"\n")
        yutils.display(OUTFILE, "\n")

        for darray in darraylist:
            for blklen in blklens:
                generate_kernels(b, darray, blklen)

        OUTFILE.close()

    ##### generate the core pack/unpack kernel declarations
    filename = "src/backend/seq/pup/yaksuri_seqi_pup.h"
    yutils.copyright_c(filename)
    OUTFILE = open(filename, "a")
    yutils.display(OUTFILE, "#ifndef YAKSURI_SEQI_PUP_H_INCLUDED\n")
    yutils.display(OUTFILE, "#define YAKSURI_SEQI_PUP_H_INCLUDED\n")
    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "#include <string.h>\n")
    yutils.display(OUTFILE, "#include <stdint.h>\n")
    yutils.display(OUTFILE, "#include \"yaksi.h\"\n")
    yutils.display(OUTFILE, "\n")

    for b in builtin_types:
        for darray in darraylist:
            for blklen in blklens:
                # individual blocklength optimization is only for
                # hvector and blkhindx
                if (len(darray) and darray[-1] != "hvector" and darray[-1] != "blkhindx" \
                    and blklen != "generic"):
                    continue

                for func in "pack","unpack":
                    ##### figure out the function name to use
                    s = "int yaksuri_seqi_%s_" % func
                    for d in darray:
                        s = s + "%s_" % d
                    # hvector and hindexed get blklen-specific function names
                    if (len(darray) and darray[-1] != "hvector" and darray[-1] != "blkhindx"):
                        s = s + b.replace(" ", "_")
                    else:
                        s = s + "blklen_%s_" % blklen + b.replace(" ", "_")
                    yutils.display(OUTFILE, "%s" % s),
                    yutils.display(OUTFILE, "(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type);\n")

    yutils.display(OUTFILE, "#endif  /* YAKSURI_SEQI_PUP_H_INCLUDED */\n")
    OUTFILE.close()

    ##### generate the switching logic to select pup functions
    gencomm.populate_pupfns(args.pup_max_nesting, "seq", blklens, builtin_types, builtin_maps)
