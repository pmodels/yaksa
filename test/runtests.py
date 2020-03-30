#! /usr/bin/env python3
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import os
import time
import argparse
import subprocess
import signal
import datetime
import xml.etree.ElementTree as ET

class colors:
    FAILURE = '\033[1;31m' # red
    SUCCESS = '\033[1;32m' # green
    INFO    = '\033[1;33m' # yellow
    PREFIX  = '\033[1;36m' # cyan
    OTHER   = '\033[1;35m' # purple
    END     = '\033[0m'    # reset

# start dir
origdir = ""

# junit variables
num_tests = 0
num_failures = 0
testnames = []
testtimes = []
testretvals = []
testoutputs = []


def printout(line, ret, elapsed_time, output):
    global num_tests
    global num_failures

    num_tests = num_tests + 1
    sys.stdout.write(colors.PREFIX + ">>>> " + colors.END)
    sys.stdout.write("return status: ")
    if (ret == 0):
        sys.stdout.write(colors.SUCCESS + "SUCCESS\n" + colors.END)
    else:
        sys.stdout.write(colors.FAILURE + "FAILURE\n" + colors.END)
        num_failures = num_failures + 1
    sys.stdout.write(colors.PREFIX + ">>>> " + colors.END)
    sys.stdout.write("elapsed time: %f sec\n" % elapsed_time)
    if (ret != 0):
        if (output != ""):
            print(colors.FAILURE + "==== execution failed with the following output ====" + colors.END)
            print(output)
            print(colors.FAILURE + "==== execution output complete ====\n" + colors.END)

    testnames.append(line)
    testtimes.append(elapsed_time)
    testretvals.append(ret)
    testoutputs.append(output)


def getlines(fh):
    alllines = fh.readlines()
    reallines = []
    for line in alllines:
        # skip comments
        if line.startswith("#"):
            continue
        # skip empty lines
        if not line.strip():
            continue
        reallines.append(line)
    return reallines


def create_summary(testlist, summary_file):
    global num_tests
    global num_failures

    tlist = open(testlist, "r")
    lines = getlines(tlist)
    tlist.close()

    # we need to make two passes on the list, one to find the number
    # of failures, and another to create the actual summary file
    for line in lines:
        # if it's a directory, absorb any summary files that might
        # exist.  otherwise, create our part of the summary
        dirname = line.split(' ', 1)[0].rstrip()
        if (os.path.isdir(dirname) and os.path.isfile(dirname + "/summary.junit.xml")):
            try:
                tree = ET.parse(dirname + "/summary.junit.xml")
            except:
                print("error parsing %s/summary.junit.xml" % dirname)
                sys.exit()
            testsuites = tree.getroot()
            testsuite = testsuites[0]
            num_failures = num_failures + int(testsuite.get('failures'))
            num_tests = num_tests + int(testsuite.get('tests'))


    # open the summary file and write to it
    try:
        fh = open(summary_file, "w")
    except:
        sys.stderr.write(colors.FAILURE + ">>>> ERROR: " + colors.END)
        sys.stderr.write("could not open summary file %s\n" % summary_file)
        sys.exit()
    fh.write("<testsuites>\n")
    fh.write("  <testsuite failures=\"%d\"\n" % num_failures)
    fh.write("             errors=\"0\"\n")
    fh.write("             skipped=\"0\"\n")
    fh.write("             tests=\"%d\"\n" % num_tests)
    fh.write("             date=\"%s\"\n" % datetime.datetime.now())
    fh.write("             name=\"summary_junit_xml\">\n")


    # second pass on the child summary files to extract their actual content
    x = 0
    for line in lines:
        # if it's a directory, absorb any summary files that might
        # exist.  otherwise, create our part of the summary
        dirname = line.split(' ', 1)[0].rstrip()
        if (os.path.isdir(dirname) and os.path.isfile(dirname + "/summary.junit.xml")):
            try:
                tree = ET.parse(dirname + "/summary.junit.xml")
            except:
                print("error parsing %s/summary.junit.xml" % dirname)
                sys.exit()
            testsuites = tree.getroot()
            testsuite = testsuites[0]
            for testcase in testsuite:
                fh.write("    <testcase name=\"%s/%s\" time=\"%s\">\n" % \
                         (dirname, testcase.get('name'), testcase.get('time')))
                for log in testcase.iter('failure'):
                    fh.write("      <failure><![CDATA[\n")
                    fh.write(log.text.strip() + "\n")
                    fh.write("      ]]></failure>\n")
                fh.write("    </testcase>\n")
        else:
            if (x < len(testnames)):
                fh.write("    <testcase name=\"%s\" time=\"%f\">\n" % (testnames[x].strip(), testtimes[x]))
                if (testretvals[x] != 0 and testoutputs[x]):
                    fh.write("      <failure><![CDATA[\n")
                    fh.write(testoutputs[x] + "\n")
                    fh.write("      ]]></failure>\n")
                fh.write("    </testcase>\n")
                x = x + 1

    fh.write("  </testsuite>\n")
    fh.write("</testsuites>\n")
    fh.close()


def wait_with_signal(p, testlist, summary_file):
    try:
        ret = p.wait()
    except:
        p.kill()
        p.wait()
        # about to die, create partial summary
        os.chdir(origdir)
        create_summary(testlist, summary_file)
        sys.exit()
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testlist', help='testlist file to execute', required=True)
    parser.add_argument('--summary', help='file to write the summary to', required=True)
    args = parser.parse_args()

    args.testlist = os.path.abspath(args.testlist)
    args.summary = os.path.abspath(args.summary)
    origdir = os.getcwd()

    try:
        fh = open(args.testlist, "r")
    except:
        sys.stderr.write(colors.FAILURE + ">>>> ERROR: " + colors.END)
        sys.stderr.write("could not open testlist %s\n" % args.testlist)
        sys.exit()

    print(colors.INFO + "\n==== executing testlist %s ====" % args.testlist + colors.END)

    lines = getlines(fh)

    firstline = 1
    for line in lines:
        if (firstline):
            firstline = 0
        else:
            print("")


        ############################################################################
        # if the first argument is a directory, step into the
        # directory and reexecute make
        ############################################################################
        dirname = line.split(' ', 1)[0].rstrip()
        if (os.path.isdir(dirname)):
            sys.stdout.write(colors.PREFIX + ">>>> " + colors.END)
            sys.stdout.write(colors.OTHER + "stepping into directory %s\n" % dirname + colors.END)
            olddir = os.getcwd()
            os.chdir(dirname)
            chdirargs = "make -s testing".split(' ')
            chdirargs = map(lambda s: s.strip(), chdirargs)
            p = subprocess.Popen(chdirargs)
            wait_with_signal(p, args.testlist, args.summary)
            os.chdir(olddir)
            continue


        # command line to process
        sys.stdout.write(line)


        ############################################################################
        # make executable
        ############################################################################
        execname = line.split(' ', 1)[0].rstrip()
        p = subprocess.Popen(['make', execname], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        ret = wait_with_signal(p, args.testlist, args.summary)
        out = p.communicate()
        if (ret != 0):
            print(colors.FAILURE + "\n==== \"make %s\" output ====" % execname + colors.END)
            print(out[0].decode().strip())
            print(colors.FAILURE + "==== make output complete ====\n" + colors.END)
            continue  # skip over to the next line


        ############################################################################
        # run the executable
        ############################################################################
        fullcmd = "./" + line
        cmdargs = fullcmd.split(' ')
        cmdargs = map(lambda s: s.strip(), cmdargs)
        start = time.time()
        p = subprocess.Popen(cmdargs, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        ret = wait_with_signal(p, args.testlist, args.summary)
        out = p.communicate()
        end = time.time()
        printout(line, ret, end - start, out[0].decode().strip())

    print(colors.INFO + "==== done executing testlist %s ====" % args.testlist + colors.END)
    create_summary(args.testlist, args.summary)

    fh.close()
