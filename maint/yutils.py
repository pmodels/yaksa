#! /usr/bin/env python
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys

########################################################################################
##### add the copyright header to the top of the file
########################################################################################
def copyright(outfile):
    OUTFILE = open(outfile, "w")
    OUTFILE.write("/*\n")
    OUTFILE.write("* Copyright (C) by Argonne National Laboratory\n")
    OUTFILE.write("*     See COPYRIGHT in top-level directory\n")
    OUTFILE.write("*\n")
    OUTFILE.write("* DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!\n")
    OUTFILE.write("*/\n")
    OUTFILE.write("\n")
    OUTFILE.close()


########################################################################################
##### generate an array of datatype arrays
########################################################################################
def generate_darrays(derived_types, darraylist, maxlevels):
    for level in range(maxlevels, 0, -1):
        index = [ ]
        for x in range(level):
            index.append(0)

        while True:
            darray = [ ]
            for x in range(level):
                darray.append(derived_types[index[x]])
            darraylist.append(darray)

            index[-1] = index[-1] + 1
            for x in range(level - 1, 0, -1):
                if (index[x] == len(derived_types)):
                    index[x] = 0
                    index[x-1] = index[x-1] + 1
            if (index[0] == len(derived_types)):
                break
