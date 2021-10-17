#############################################################
#                                                           #
#      TDT4200 - Image Morphing Exercise GUI                #
#                                                           #
#      Author: Tor Andre Haugdahl <torahaug@ntnu.no>        #
#                                                           #
#############################################################


.PHONY: print
.PHONY: clean

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PARENT_DIR := $(shell dirname $(ROOT_DIR))

CC         := g++
IMPL_EXT   := cpp
HEADER_EXT := hpp

# PP_DIRECTIVES := -DDEBUG
PP_DIRECTIVES :=


# TODO: Add support for Mac OS (Darwin)
LDFLAGS = -lGL -lm -lglfw -lGLEW
CFLAGS = -g -I${ROOT_DIR}/inc -I${PARENT_DIR}/includes/glm -I${PARENT_DIR}/includes ${PP_DIRECTIVES}

PROJECT_NAME = gmorph

SRC_FILES = $(wildcard src/*.${IMPL_EXT})
OBJ_FILES = $(patsubst src/%.${IMPL_EXT}, build/%.o, ${SRC_FILES})

${PROJECT_NAME}: build/${PROJECT_NAME}.o $(OBJ_FILES)
	$(CC) -o ${PROJECT_NAME} ${CFLAGS} ${LDFLAGS} ${OBJ_FILES}

build/%.o: src/%.${IMPL_EXT}
	$(CC) -c $(CFLAGS) -o $@ $<

print:
	echo ${SRC_FILES}
	echo ${PROJECT_DIRECTORY}

clean:
	cd build && rm *
