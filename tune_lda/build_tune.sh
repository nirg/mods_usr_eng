#!/bin/bash
# tune
cythonize _tune.pyx
gcc -fno-strict-aliasing -I/Users/nirg/anaconda/include -arch x86_64 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/nirg/anaconda/include/python2.7 -c _tune.c -o _tune.o
gcc -fno-strict-aliasing -I/Users/nirg/anaconda/include -arch x86_64 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/nirg/anaconda/include/python2.7 -c gamma.c -o gamma.o
gcc -bundle -undefined dynamic_lookup -L/Users/nirg/anaconda/lib -arch x86_64 -arch x86_64 _tune.o gamma.o -L/Users/nirg/anaconda/lib -o _tune.so
# lda
cythonize _lda.pyx
gcc -fno-strict-aliasing -I/Users/nirg/anaconda/include -arch x86_64 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/nirg/anaconda/include/python2.7 -c _lda.c -o _lda.o
gcc -bundle -undefined dynamic_lookup -L/Users/nirg/anaconda/lib -arch x86_64 -arch x86_64 _lda.o gamma.o -L/Users/nirg/anaconda/lib -o _lda.so
