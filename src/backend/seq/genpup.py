#! /usr/bin/env python
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys

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
def hvector_decl(nesting, dtp, b):
    display("int count%d = %s->u.hvector.count;\n" % (nesting, dtp))
    display("int blocklength%d ATTRIBUTE((unused)) = %s->u.hvector.blocklength;\n" % (nesting, dtp))
    display("intptr_t stride%d = %s->u.hvector.stride / sizeof(%s);\n" % (nesting, dtp, b))
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def hvector(suffix, b, blklen, last):
    global indentation
    global num_paren_open
    num_paren_open += 2
    display("for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    indentation += 1
    if (blklen == "generic"):
        display("for (int k%d = 0; k%d < blocklength%d; k%d++) {\n" % (suffix, suffix, suffix, suffix))
    else:
        display("for (int k%d = 0; k%d < %s; k%d++) {\n" % (suffix, suffix, blklen, suffix))
    indentation += 1
    global s
    if (last != 1):
        s += " + j%d * stride%d + k%d * extent%d" % (suffix, suffix, suffix, suffix + 1)
    else:
        s += " + j%d * stride%d + k%d" % (suffix, suffix, suffix)

## blkhindx routines
def blkhindx_decl(nesting, dtp, b):
    display("int count%d = %s->u.blkhindx.count;\n" % (nesting, dtp))
    display("int blocklength%d ATTRIBUTE((unused)) = %s->u.blkhindx.blocklength;\n" % (nesting, dtp))
    display("intptr_t *restrict array_of_displs%d = %s->u.blkhindx.array_of_displs;\n" % (nesting, dtp))
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def blkhindx(suffix, b, blklen, last):
    global indentation
    global num_paren_open
    num_paren_open += 2
    display("for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    indentation += 1
    if (blklen == "generic"):
        display("for (int k%d = 0; k%d < blocklength%d; k%d++) {\n" % (suffix, suffix, suffix, suffix))
    else:
        display("for (int k%d = 0; k%d < %s; k%d++) {\n" % (suffix, suffix, blklen, suffix))
    indentation += 1
    global s
    if (last != 1):
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d * extent%d" % \
             (suffix, suffix, b, suffix, suffix + 1)
    else:
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d" % (suffix, suffix, b, suffix)

## hindexed routines
def hindexed_decl(nesting, dtp, b):
    display("int count%d = %s->u.hindexed.count;\n" % (nesting, dtp))
    display("int *restrict array_of_blocklengths%d = %s->u.hindexed.array_of_blocklengths;\n" % (nesting, dtp))
    display("intptr_t *restrict array_of_displs%d = %s->u.hindexed.array_of_displs;\n" % (nesting, dtp))
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def hindexed(suffix, b, blklen, last):
    global indentation
    global num_paren_open
    num_paren_open += 2
    display("for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    indentation += 1
    display("for (int k%d = 0; k%d < array_of_blocklengths%d[j%d]; k%d++) {\n" % \
            (suffix, suffix, suffix, suffix, suffix))
    indentation += 1
    global s
    if (last != 1):
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d * extent%d" % \
             (suffix, suffix, b, suffix, suffix + 1)
    else:
        s += " + array_of_displs%d[j%d] / sizeof(%s) + k%d" % (suffix, suffix, b, suffix)

## dup routines
def dup_decl(nesting, dtp, b):
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def dup(suffix, b, blklen, last):
    pass

## contig routines
def contig_decl(nesting, dtp, b):
    display("int count%d = %s->u.contig.count;\n" % (nesting, dtp))
    display("intptr_t stride%d = %s->u.contig.child->extent / sizeof(%s);\n" % (nesting, dtp, b))
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def contig(suffix, b, blklen, last):
    global indentation
    global num_paren_open
    num_paren_open += 1
    display("for (int j%d = 0; j%d < count%d; j%d++) {\n" % (suffix, suffix, suffix, suffix))
    indentation += 1
    global s
    s += " + j%d * stride%d" % (suffix, suffix)

# resized routines
def resized_decl(nesting, dtp, b):
    display("uintptr_t extent%d ATTRIBUTE((unused)) = %s->extent / sizeof(%s);\n" % (nesting, dtp, b))

def resized(suffix, b, blklen, last):
    pass


## loop through the derived and basic types to generate individual
## pack functions
builtin_types = [ "char", "wchar_t", "int", "short", "long", "long long", "int8_t", "int16_t", \
                  "int32_t", "int64_t", "float", "double", "long double" ]
derived_types = [ "hvector", "blkhindx", "hindexed", "dup", "contig", "resized", "" ]
derived_maps = {
    "hvector": hvector,
    "blkhindx": blkhindx,
    "hindexed": hindexed,
    "dup": dup,
    "contig": contig,
    "resized": resized,
}
derived_decl_maps = {
    "hvector": hvector_decl,
    "blkhindx": blkhindx_decl,
    "hindexed": hindexed_decl,
    "dup": dup_decl,
    "contig": contig_decl,
    "resized": resized_decl,
}
blklens = [ "1", "2", "3", "4", "5", "6", "7", "8", "generic" ]


########################################################################################
##### Basic type specific functions
########################################################################################
for b in builtin_types:
    OUTFILE = open("src/backend/seq/pup/yaksuri_seqi_pup_%s.c" % b.replace(" ","_"), "w")
    OUTFILE.write("/*\n")
    OUTFILE.write("* Copyright (C) by Argonne National Laboratory\n")
    OUTFILE.write("*     See COPYRIGHT in top-level directory\n")
    OUTFILE.write("*\n")
    OUTFILE.write("* DO NOT EDIT: AUTOMATICALLY GENERATED BY genpup.py\n")
    OUTFILE.write("*/\n")
    OUTFILE.write("\n")
    OUTFILE.write("#include <string.h>\n")
    OUTFILE.write("#include <stdint.h>\n")
    OUTFILE.write("#include <wchar.h>\n")
    OUTFILE.write("#include \"yaksuri_seqi_pup.h\"\n")
    OUTFILE.write("\n")

    for d1 in derived_types:
        for d2 in derived_types:
            if (d2 == "" and d1 != ""):
                continue
            for d3 in derived_types:
                if (d3 == "" and d2 != ""):
                    continue
                for blklen in blklens:

                    # individual blocklength optimization is only for
                    # hvector and blkhindx
                    if (d3 != "hvector" and d3 != "blkhindx" and blklen != "generic"):
                        continue

                    for func in "pack","unpack":

                        ##### figure out the function name to use
                        s = "int yaksuri_seqi_%s_" % func
                        if (d1 != ""):
                            s = s + "%s_" % d1
                        if (d2 != ""):
                            s = s + "%s_" % d2
                        if (d3 != ""):
                            s = s + "%s_" % d3
                        # hvector and hindexed get blklen-specific function names
                        if (d3 != "hvector" and d3 != "blkhindx"):
                            s = s + b.replace(" ", "_")
                        else:
                            s = s + "blklen_%s_" % blklen + b.replace(" ", "_")
                        OUTFILE.write("%s" % s),
                        OUTFILE.write("(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type, yaksi_request_s **request)\n")
                        OUTFILE.write("{\n")


                        ##### variable declarations
                        indentation += 1

                        # generic variables
                        display("int rc = YAKSA_SUCCESS;\n");
                        display("const %s *restrict sbuf = (const %s *) inbuf;\n" % (b, b));
                        display("%s *restrict dbuf = (%s *) outbuf;\n" % (b, b));
                        display("uintptr_t extent ATTRIBUTE((unused)) = type->extent / sizeof(%s);\n" % b)
                        OUTFILE.write("\n");

                        # variables specific to each nesting level
                        s = "type"
                        if (d1 != ""):
                            derived_decl_maps[d1](1, s, b)
                            OUTFILE.write("\n")
                            s = s + "->u.%s.child" % d1

                        if (d2 != ""):
                            derived_decl_maps[d2](2, s, b)
                            OUTFILE.write("\n")
                            s = s + "->u.%s.child" % d2

                        if (d3 != ""):
                            derived_decl_maps[d3](3, s, b)
                            OUTFILE.write("\n")


                        ##### shortcut for builtin functions
                        if (d1 == "" and d2 == "" and d3 == ""):
                            display("memcpy(dbuf, sbuf, count * sizeof(%s));\n" % b)
                            OUTFILE.write("\n");
                            display("return rc;\n")
                            indentation -= 1
                            OUTFILE.write("}\n\n")
                            continue


                        ##### non-hvector and non-blkhindx
                        display("uintptr_t idx = 0;\n")
                        display("for (int i = 0; i < count; i++) {\n")
                        num_paren_open += 1
                        indentation += 1
                        s = "i * extent"
                        if (d1 != ""):
                            derived_maps[d1](1, b, "generic", 0)
                        if (d2 != ""):
                            derived_maps[d2](2, b, "generic", 0)
                        if (d3 != ""):
                            derived_maps[d3](3, b, blklen, 1)
                        if (func == "pack"):
                            display("dbuf[idx++] = sbuf[%s];\n" % s)
                        else:
                            display("dbuf[%s] = sbuf[idx++];\n" % s)
                        for x in range(num_paren_open):
                            indentation -= 1
                            display("}\n")
                        num_paren_open = 0
                        OUTFILE.write("\n");
                        display("return rc;\n")
                        indentation -= 1
                        OUTFILE.write("}\n\n")



########################################################################################
##### Primary C file
########################################################################################
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
    "YAKSA_TYPE__C_COMPLEX": "float",
    "YAKSA_TYPE__C_DOUBLE_COMPLEX": "double",
    "YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX": "long_double",
    "YAKSA_TYPE__BYTE": "int8_t"
}

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

    if (t == "hvector" or t == "blkhindx"):
        display("switch (%s->u.%s.blocklength) {\n" % (child_type_str(typelist), t))
        indentation += 1
        for blklen in blklens:
            if (blklen != "generic"):
                display("case %s:\n" % blklen)
            else:
                display("default:\n")
            indentation += 1
            display("if (max_nesting_level >= %d) {\n" % nesting_level)
            display("    type->backend_priv.seq.pack = yaksuri_seqi_%s_blklen_%s_%s;\n" % (pupstr, blklen, val))
            display("    type->backend_priv.seq.unpack = yaksuri_seqi_un%s_blklen_%s_%s;\n" % (pupstr, blklen, val))
            display("}\n")
            display("break;\n")
            indentation -= 1
        indentation -= 1
        display("}\n")
    else:
        display("if (max_nesting_level >= %d) {\n" % nesting_level)
        display("    type->backend_priv.seq.pack = yaksuri_seqi_%s_%s;\n" % (pupstr, val))
        display("    type->backend_priv.seq.unpack = yaksuri_seqi_un%s_%s;\n" % (pupstr, val))
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
        if (d == ""):
            display("case YAKSI_TYPE_KIND__BUILTIN:\n")
            indentation += 1
            switcher_builtin(typelist, pupstr)
            display("break;\n")
            indentation -= 1
        elif (nests > 1):
            display("case YAKSI_TYPE_KIND__%s:\n" % d.upper())
            indentation += 1
            typelist.append(d)
            switcher(typelist, pupstr + "_%s" % d, nests - 1)
            typelist.pop()
            display("break;\n")
            indentation -= 1
        indentation -= 1

    indentation += 1
    display("default:\n")
    display("    break;\n")
    indentation -= 1
    display("}\n")


OUTFILE = open("src/backend/seq/pup/yaksuri_seqi_pup.c", "w")
OUTFILE.write("\
/*\n\
 * Copyright (C) by Argonne National Laboratory\n\
 *     See COPYRIGHT in top-level directory\n\
 *\n\
 * DO NOT EDIT: AUTOMATICALLY GENERATED BY genpup.py\n\
 */\n\
\n\
#include <stdio.h>\n\
#include <stdlib.h>\n\
#include <wchar.h>\n\
#include \"yaksi.h\"\n\
#include \"yaksu.h\"\n\
#include \"yaksuri_seqi.h\"\n\
#include \"yaksuri_seqi_pup.h\"\n\
\n\
int yaksuri_seqi_populate_pupfns(yaksi_type_s * type)\n\
{\n\
    int rc = YAKSA_SUCCESS;\n\
\n\
    type->backend_priv.seq.pack = NULL;\n\
    type->backend_priv.seq.unpack = NULL;\n\
\n\
    char *str = getenv(\"YAKSA_ENV_MAX_NESTING_LEVEL\");\n\
    int max_nesting_level;\n\
    if (str) {\n\
        max_nesting_level = atoi(str);\n\
    } else {\n\
        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;\n\
    }\n\
\n\
");

indentation += 1
pupstr = "pack"
typelist = [ ]
switcher(typelist, pupstr, 4)
OUTFILE.write("\n")
display("return rc;\n")
indentation -= 1
display("}\n")

OUTFILE.close()


########################################################################################
##### Primary header file
########################################################################################
OUTFILE = open("src/backend/seq/pup/yaksuri_seqi_pup.h", "w")
OUTFILE.write("/*\n")
OUTFILE.write("* Copyright (C) by Argonne National Laboratory\n")
OUTFILE.write("*     See COPYRIGHT in top-level directory\n")
OUTFILE.write("*\n")
OUTFILE.write("* DO NOT EDIT: AUTOMATICALLY GENERATED BY genpup.py\n")
OUTFILE.write("*/\n")
OUTFILE.write("\n")
OUTFILE.write("#ifndef YAKSURI_SEQI_PUP_H_INCLUDED\n")
OUTFILE.write("#define YAKSURI_SEQI_PUP_H_INCLUDED\n")
OUTFILE.write("\n")
OUTFILE.write("#include <string.h>\n")
OUTFILE.write("#include <stdint.h>\n")
OUTFILE.write("#include \"yaksi.h\"\n")
OUTFILE.write("\n")

for b in builtin_types:
    for d1 in derived_types:
        for d2 in derived_types:
            if (d2 == "" and d1 != ""):
                continue
            for d3 in derived_types:
                if (d3 == "" and d2 != ""):
                    continue
                for blklen in blklens:

                    # individual blocklength optimization is only for
                    # hvector and blkhindx
                    if (d3 != "hvector" and d3 != "blkhindx" and blklen != "generic"):
                        continue

                    for func in "pack","unpack":
                        ##### figure out the function name to use
                        s = "int yaksuri_seqi_%s_" % func
                        if (d1 != ""):
                            s = s + "%s_" % d1
                        if (d2 != ""):
                            s = s + "%s_" % d2
                        if (d3 != ""):
                            s = s + "%s_" % d3
                        # hvector and hindexed get blklen-specific function names
                        if (d3 != "hvector" and d3 != "blkhindx"):
                            s = s + b.replace(" ", "_")
                        else:
                            s = s + "blklen_%s_" % blklen + b.replace(" ", "_")
                        OUTFILE.write("%s" % s),
                        OUTFILE.write("(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type, yaksi_request_s **request);\n")

## end of basic-type specific file
OUTFILE.write("#endif  /* YAKSURI_SEQI_PUP_H_INCLUDED */\n")
OUTFILE.close()
