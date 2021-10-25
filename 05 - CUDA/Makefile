.PHONY: clean
.PHONY: print

######################
## GLOBAL VARIABLES ##
######################
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


##########################
## SERIAL MORPH PROGRAM ##
##########################
SERIAL_CC:=nvcc

main-serial: main_serial.cu
	$(SERIAL_CC) $^ -o $@


############################
## PARALLEL MORPH PROGRAM ##
############################
PARALLEL_CC:=nvcc

main: main_solution.cu
	$(PARALLEL_CC) $^ -o $@

print:
	echo $(ROOT_DIR)
	echo $(SERIAL_OBJ_FILES)
