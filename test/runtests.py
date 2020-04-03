#! /usr/bin/env python3
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import os
import re
import time
import argparse
import subprocess
import signal
import datetime

opts = {'verbose': 1}

class colors:
    FAILURE = '\033[1;31m' # red
    SUCCESS = '\033[1;32m' # green
    INFO    = '\033[1;33m' # yellow
    PREFIX  = '\033[1;36m' # cyan
    OTHER   = '\033[1;35m' # purple
    END     = '\033[0m'    # reset

def init_colors():
    if not sys.stdout.isatty():
        colors.FAILURE = ''
        colors.SUCCESS = ''
        colors.INFO = ''
        colors.PREFIX = ''
        colors.OTHER = ''
        colors.END = ''

def printout(line, ret, elapsed_time, output):
    sys.stdout.write(colors.PREFIX + ">>>> " + colors.END)
    sys.stdout.write("return status: ")
    if (ret == 0):
        sys.stdout.write(colors.SUCCESS + "SUCCESS\n" + colors.END)
    else:
        sys.stdout.write(colors.FAILURE + "FAILURE\n" + colors.END)
    sys.stdout.write(colors.PREFIX + ">>>> " + colors.END)
    sys.stdout.write("elapsed time: %f sec\n" % elapsed_time)
    if (ret != 0):
        if (output != ""):
            print(colors.FAILURE + "==== execution failed with the following output ====" + colors.END)
            print(output)
            print(colors.FAILURE + "==== execution output complete ====\n" + colors.END)


def create_summary(tests, summary_file):
    num_tests = 0
    num_failures = 0

    # first pass to get the stat
    for test in tests:
        num_tests += 1
        if test['ret'] != 0 and test['out']:
            num_failures += 1

    print("tests run: %d, failed: %d" % (num_tests, num_failures))
    print("Writing summary: " + summary_file)
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


    # second pass for actual content
    for test in tests:
        fh.write("    <testcase name=\"%s\" time=\"%f\">\n" % (test['dir'] + '/' + test['line'].strip(), test['time']))
        if test['ret'] != 0 and test['out']:
            fh.write("      <failure><![CDATA[\n")
            fh.write(test['out'] + "\n")
            fh.write("      ]]></failure>\n")
        fh.write("    </testcase>\n")

    fh.write("  </testsuite>\n")
    fh.write("</testsuites>\n")
    fh.close()


def wait_with_signal(p, test):
    try:
        ret = p.wait()
    except:
        p.kill()
        p.wait()
        # about to die, create partial summary
        test['result'] = 'fail'
    return ret

def load_tests(testlist):
    tests = []
    tests.append({'line': "echo executing testlist " + testlist})

    test_dir = os.path.dirname(testlist)
    if test_dir == '':
        test_dir = '.'

    test_prefix = '.'

    try:
        fh = open(testlist, "r")
    except:
        sys.stderr.write(colors.FAILURE + ">>>> ERROR: " + colors.END)
        sys.stderr.write("could not open testlist %s\n" % testlist)
        sys.exit()

    for line in fh:
        if not line.strip() or line.startswith("#"):
            continue

        m = re.match(r'(\w+):\s*(\S+)', line)
        if m:
            # testlist: condition: etc.
            if m.group(1) == 'testlist':
                sub_tests = load_tests(test_dir + '/' + m.group(2))
                tests.extend(sub_tests)
            elif m.group(1) == 'condition':
                if not os.environ.get(m.group(2)):
                    # condition not found, skip this testlist
                    return []
            elif m.group(1) == 'prefix':
                test_prefix = m.group(2)
        else:
            # test to be run
            tests.append({'dir': test_prefix, 'line': line})
    fh.close()
    tests.append({'line': "echo done executing testlist " + testlist})
    return tests

def run_test(test):
    m = re.match(r'echo\s+(.*)', test['line'])
    if m:
        print(colors.INFO + m.group(1) + colors.END)
        test['type'] = 'info'
    else:
        test['type'] = 'test'
        test_dir = test['dir']
        line = test['line']
        m = re.match(r'(\S+)(.*)', test['line'])
        execname = test_dir + '/' + m.group(1)

        print(line)

        ############################################################################
        # make executable
        ############################################################################
        if opts['verbose']:
            print('make ' + execname)
        p = subprocess.Popen(['make', execname], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        ret = wait_with_signal(p, test)
        out = p.communicate()
        if opts['verbose']:
            print(out[0].decode().strip())
        if (ret != 0):
            print(colors.FAILURE + "\n==== \"make %s\" output ====" % execname + colors.END)
            print(out[0].decode().strip())
            print(colors.FAILURE + "==== make output complete ====\n" + colors.END)
            test['stage'] = 'make'
            test['time'] = 0
            test['out'] = out[0].decode().strip()
            test['ret'] = ret
            return


        ############################################################################
        # run the executable
        ############################################################################
        origdir = os.getcwd()
        os.chdir(test_dir)
        fullcmd = "./" + line
        if opts['verbose']:
            print(fullcmd)
        cmdargs = fullcmd.split(' ')
        cmdargs = map(lambda s: s.strip(), cmdargs)
        start = time.time()
        p = subprocess.Popen(cmdargs, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        ret = wait_with_signal(p, test)
        out = p.communicate()
        end = time.time()
        printout(line, ret, end - start, out[0].decode().strip())
        test['time'] = end - start
        test['ret'] = ret
        test['out'] = out[0].decode().strip()
        os.chdir(origdir)


if __name__ == '__main__':
    init_colors()

    parser = argparse.ArgumentParser()
    parser.add_argument('testlists', help='testlist files to execute', nargs='+')
    parser.add_argument('--summary', help='file to write the summary to', required=True)
    args = parser.parse_args()

    args.summary = os.path.abspath(args.summary)

    tests = []
    for testlist in args.testlists:
        tests.extend(load_tests(os.path.abspath(testlist)))

    firstline = 1
    for test in tests:
        if (firstline):
            firstline = 0
        else:
            print("")

        run_test(test)

    # prune away non-tests
    tests[:] = [test for test in tests if test['type'] == 'test']

    create_summary(tests, args.summary)
