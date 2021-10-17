#ifndef _MORPHFILE_HPP_
#define _MORPHFILE_HPP_


#include <FeatureLine.hpp>
#include <string>

enum {
    MORPH_FILE_READ_NUM_PAIRS_MISSING,
    MORPH_FILE_READ_INCORRECT_PAIR,
    MORPH_FILE_READ_SUCCESS
};

/**
 * Contains the functionality for reading and writing the morph configuration file.
 */
class MorphFile
{
public:
    MorphFile() = delete;
    MorphFile(const std::string &path);

    ~MorphFile();

    void read(FeatureLineManager *fl_manager);

    void write(FeatureLineManager *fl_manager);

private:
    std::string filename;

};

class MorphFileReader
{

public:
    MorphFileReader() = delete;

    static void read(const std::string &path, FeatureLineManager *fl_manager);

private:
    static int readPair(FILE *fh, FeatureLine **first, FeatureLine **second);

    static int readNumPairs(FILE *fh, int *numPairs);

private:


};

struct MorphFileWriter
{
    static void write(const std::string &path, FeatureLineManager *fl_manager);
};


#endif
