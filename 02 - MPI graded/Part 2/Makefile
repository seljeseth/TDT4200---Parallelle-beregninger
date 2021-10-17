.PHONY: clean
.PHONY: print

######################
## GLOBAL VARIABLES ##
######################
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# replace " " with "\ " in path
null :=
space := ${null} ${null}
ROOT_DIR:=$(subst $(space),\ ,$(CURDIR))

##########################
## SERIAL MORPH PROGRAM ##
##########################
SERIAL_CC:=gcc
SERIAL_FLAGS:=-lm -g
SERIAL_INCLUDE_DIRS:=-I$(ROOT_DIR)/inc-serial -I$(ROOT_DIR)/inc

SERIAL_SRC_FILES:=$(wildcard src-serial/*.c)
SERIAL_OBJ_FILES:=$(patsubst src-serial/%.c,build-serial/%.o,$(SERIAL_SRC_FILES))

build-serial/%.o: src-serial/%.c
	$(SERIAL_CC) $< $(SERIAL_FLAGS) $(SERIAL_INCLUDE_DIRS) -c -o $@

main-serial: $(SERIAL_OBJ_FILES)
	$(SERIAL_CC) $^ $(SERIAL_FLAGS) $(SERIAL_INCLUDE_DIRS) -o $@


############################
## PARALLEL MORPH PROGRAM ##
############################
PARALLEL_CC:=mpicc
PARALLEL_FLAGS:=-lm -g

PARALLEL_SRC_FILES:=$(wildcard src/*.c)
PARALLEL_OBJ_FILES:=$(patsubst src/%.c,build/%.o,$(PARALLEL_SRC_FILES))

PARALLEL_INCLUDE_PATHS:=-I$(ROOT_DIR)/inc

build/%.o: src/%.c
	$(PARALLEL_CC) $< $(PARALLEL_FLAGS) $(PARALLEL_INCLUDE_PATHS) -c -o $@

main: $(PARALLEL_OBJ_FILES)
	$(PARALLEL_CC) $^ $(PARALLEL_FLAGS) -o $@


print:
	echo $(ROOT_DIR)
	echo $(SERIAL_OBJ_FILES)
