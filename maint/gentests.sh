#! /bin/bash
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

echo_n() {
    # "echo -n" isn't portable, must portably implement with printf
    printf "%s" "$*"
}

add_to_testlist() {
    echo "$@" >> ${testlist}
}

counts=(17 1018 1024 65530 1048576)

# generate test list
echo "=== generating test list ==="
seed=1

#### SIMPLE tests
testlist=test/simple/testlist
rm -f ${testlist}
echo_n "generating simple tests ... "
add_to_testlist "simple_test"
add_to_testlist "threaded_test"
echo "done"


#### PACK tests
testlist=test/pack/testlist
rm -f ${testlist}
echo_n "generating pack tests ... "
add_to_testlist "# pack tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 1 -ordering normal -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# reverse offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering reverse -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# random offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering random -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# overlapping packing/unpacking tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap regular"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# irregular overlapping packing/unpacking tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "pack -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap irregular"
	seed=$((seed + 1))
    done
done
echo "done"


#### IOV tests
testlist=test/iov/testlist
rm -f ${testlist}
echo_n "generating iov tests ... "
add_to_testlist "# iov tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 1 -ordering normal -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# reverse offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering reverse -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# random offset tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering random -overlap none"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# overlapping ioving/unioving tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap regular"
	seed=$((seed + 1))
    done
done

add_to_testlist
add_to_testlist "# irregular overlapping ioving/unioving tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "iov -datatype $t -count $count -seed $seed -iters $iters -segments 64 -ordering normal -overlap irregular"
	seed=$((seed + 1))
    done
done
echo "done"


#### Flatten tests
testlist=test/flatten/testlist
rm -f ${testlist}
echo_n "generating flatten tests ... "
add_to_testlist "# flatten tests"
for count in ${counts[@]} ; do
    for t in int char double double_int short_int int:2,double:3 int:3,double:2 ; do
	if test "$count" -le "1024" ; then
	    iters=32768
	else
	    iters=128
	fi
	add_to_testlist "flatten -datatype $t -count $count -seed $seed -iters $iters"
	seed=$((seed + 1))
    done
done
echo "done"

echo "=== done === "
