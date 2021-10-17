#include <ArgumentParser.hpp>
#include <cstdio>
#include <memory.h>


void print_usage_info()
{
    printf("Image Morphing GUI\n");
    printf("================================\n");
    printf("Usage: ./gmorph path-to-source-image path-to-destination-image [featureline-file]\n");
    printf("featureline-file defaults to \"config\"\n");
}

Arguments *parse_args(int argc, char **argv)
{
    if(argc < 3)
    {
        abort();
    }

    Arguments *args = (Arguments *) malloc(sizeof(Arguments));

    args->source_img = new std::string{argv[1]};
    args->dest_img = new std::string{argv[2]};
    args->featureline_file = nullptr;

    if(argc >= 4) {
        args->featureline_file = new std::string{argv[3]};
    }

    return args;
}
