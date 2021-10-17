#ifndef _ARGUMENTPARSER_HPP_
#define _ARGUMENTPARSER_HPP_

#include <string>

/**
 * This struct contains all supported arguments
 */
struct Arguments
{
    std::string *source_img;
    std::string *dest_img;
    std::string *featureline_file;
};

Arguments *parse_args(int argc, char **argv);
void print_usage_info();


#endif
