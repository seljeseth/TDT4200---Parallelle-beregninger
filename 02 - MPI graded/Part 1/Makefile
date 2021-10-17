.PHONY: clean
.PHONY: print

BUILD_DIR = build
create_folder := $(shell mkdir -p $(BUILD_DIR))
   
######################
## GLOBAL VARIABLES ##
######################

# replace " " with "\ " in path
null :=
space := ${null} ${null}
ROOT_DIR:=$(subst $(space),\ ,$(CURDIR))


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
	ROOT_DIR=$(ROOT_DIR)

clean:
	rm -rf build
