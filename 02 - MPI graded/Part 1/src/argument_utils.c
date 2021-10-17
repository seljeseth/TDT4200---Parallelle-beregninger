#include <argument_utils.h>
#include <getopt.h>
#include <stddef.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>


OPTIONS *parse_args(int argc, char **argv)
{
   /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex = 2;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"kernel",     required_argument, 0, 'k'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hk:i:";
  {
    char *endptr;
    int c;
    int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        // goto graceful_exit;
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxKernelIndex) {
          help(argv[0], c, optarg);
          return NULL;
        }
        kernelIndex = (unsigned int) parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          return NULL;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    return NULL;
  }

  unsigned int arglen = strlen(argv[optind]);
  input = calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  OPTIONS* result = malloc(sizeof(OPTIONS));
  result->iterations = iterations;
  result->output = output;
  result->input = input;
  result->kernelIndex = kernelIndex;
  result->ret = ret;

  return result;
}


void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -k, --kernel     <kernel>        kernel index (0<=x<=%u) (2)\n", maxKernelIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}
