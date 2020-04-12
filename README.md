# ParallelFramework

## master branch
* windows 7 64bit
* cuda 10.1
* visual studio 2019
* visual studio build tools
* Intel MPI, I_MPI_ROOT environment variable set
* #### USELESS

## linux branch
* Ubuntu 19.10
* nvcc,cuda 10.2.89
* openmpi 4.0.3
* LD_LIBRARY_PATH=/usr/local/lib  _(gia to libopen-rte)_
* gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~19.10)  _(giati o nvcc gkriniazei me >=8)_
* compile `make`/`make out`
* run `make run`

#### Gia remote configuration
* server `make run`
* clients `make run-remote`
* swap comments `Makefile:7-8`
* swap comments `framework.cu:113-114`
* swap comments `framework.h:264-265`

#### framework.run logic
1. for each gpu+1cpu spawn mpi process
2. main process: masterThread() (data distribution/collecting) & listeningThread() (socket connections handler)
3. spawned processes: (connect through socket?), slaveThread(), exit()

## Isxuoun gia commit [fe380c2](https://github.com/chrisk44/ParallelFramework/tree/fe380c22d353dbd427224cad9d429b5c3225ee30)
