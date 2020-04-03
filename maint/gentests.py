#! /usr/bin/env python3
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import re


##### global settings
counts = [ 17, 1075, 65536 ]
types = [ "int", "short_int", "int:3,double:2" ]
seed = 1


##### simple tests generator
def gen_simple_tests(testlist):
    global seed

    try:
        outfile = open(testlist, "w")
    except:
        sys.stderr.write("error creating testlist %s\n" % testlist)
        sys.exit()

    sys.stdout.write("generating simple tests ... ")
    outfile.write("prefix: test/simple\n\n")
    outfile.write("simple_test\n")
    outfile.write("threaded_test\n")
    outfile.close()
    sys.stdout.write("done\n")


##### pack/iov tests generator
def create_testlist(testlist):
    try:
        outfile = open(testlist, "w")
    except:
        sys.stderr.write("error creating testlist %s\n" % testlist)
        sys.exit()

    m = re.match(r'(test/\w+)', testlist)
    outfile.write("prefix: " + m.group(1) + "\n\n")

    if re.match(r'.*cuda.gen', testlist):
        outfile.write("condition: HAS_CUDA" + "\n\n")
    outfile.close()

def gen_pack_iov_tests(fn, testlist, create):
    global seed

    segments = [ 1, 64 ]
    orderings = [ "normal", "reverse", "random" ]
    overlaps = [ "none", "regular", "irregular" ]

    try:
        if (create == "create"):
            outfile = open(testlist, "w")
        else:
            outfile = open(testlist, "a")
    except:
        sys.stderr.write("error creating testlist %s\n" % testlist)
        sys.exit()

    sys.stdout.write("generating %s tests ... " % fn)
    if create == "create":
        m = re.match(r'(test/\w+)', testlist)
        outfile.write("prefix: " + m.group(1) + "\n\n")
    for overlap in overlaps:
        for ordering in orderings:
            if (overlap != "none" and ordering != "normal"):
                continue

            for segment in segments:
                if (segment == 1 and (ordering != "normal" or overlap != "none")):
                    continue

                outfile.write("# %s (segments %d, pack order %s, overlap %s)\n" % \
                              (fn, segment, ordering, overlap))
                for count in counts:
                    for t in types:

                        if (count <= 1024):
                            iters = 32768
                        else:
                            iters = 128

                        outstr = fn + " "
                        outstr += "-datatype %s " % t
                        outstr += "-count %d " % count
                        outstr += "-seed %d " % seed
                        seed = seed + 1
                        outstr += "-iters %d " % iters
                        outstr += "-segments %d " % segment
                        outstr += "-ordering %s " % ordering
                        outstr += "-overlap %s" % overlap
                        outfile.write(outstr + "\n")
                outfile.write("\n")
    outfile.close()
    sys.stdout.write("done\n")


##### flatten tests generator
def gen_flatten_tests(testlist):
    global seed

    try:
        outfile = open(testlist, "w")
    except:
        sys.stderr.write("error creating testlist %s\n" % testlist)
        sys.exit()

    sys.stdout.write("generating flatten tests ... ")
    outfile.write("prefix: test/flatten\n\n")
    for count in counts:
        for t in types:

            if (count <= 1024):
                iters = 32768
            else:
                iters = 128

            outstr = "flatten "
            outstr += "-datatype %s " % t
            outstr += "-count %d " % count
            outstr += "-seed %d " % seed
            seed = seed + 1
            outstr += "-iters %d" % iters
            outfile.write(outstr + "\n")
    outfile.write("\n")
    outfile.close()
    sys.stdout.write("done\n")


##### main function
if __name__ == '__main__':
    gen_simple_tests("test/simple/testlist.gen")

    gen_pack_iov_tests("pack", "test/pack/testlist.gen", "create")
    create_testlist("test/pack/testlist.cuda.gen")
    # gen_pack_iov_tests("pack_cuda_sbuf_tbuf", "test/pack/testlist.cuda.gen", "create")
    # gen_pack_iov_tests("pack_cuda_dbuf_tbuf", "test/pack/testlist.cuda.gen", "append")
    gen_pack_iov_tests("pack_cuda_sbuf_dbuf_tbuf", "test/pack/testlist.cuda.gen", "append")
    gen_pack_iov_tests("iov", "test/iov/testlist.gen", "create")

    gen_flatten_tests("test/flatten/testlist.gen")
