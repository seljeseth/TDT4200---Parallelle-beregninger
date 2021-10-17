#include <MorphFile.hpp>
#include <fstream>
#include <sstream>


MorphFile::MorphFile(const std::string &path)
{
    filename = path;
}

MorphFile::~MorphFile()
{ }


void MorphFile::read(FeatureLineManager *fl_manager)
{
    MorphFileReader::read(filename, FeatureLineManager::getInstance());
}

void MorphFile::write(FeatureLineManager *fl_manager)
{

    MorphFileWriter::write(filename, FeatureLineManager::getInstance());
}


void MorphFileReader::read(const std::string &path, FeatureLineManager *fl_manager)
{

    FILE* fh = fopen(path.c_str(), "r");

    if( ! fh ) return;

    int num_pairs;

    int readResult = readNumPairs(fh, &num_pairs);


    if ( MORPH_FILE_READ_NUM_PAIRS_MISSING == readResult )
    {
        fprintf(stderr, "The morph configuration is incorrectly formatted!\n");
        fprintf(stderr, "Reason: Number of feature line pairs is not present\n");
    }

    for(int i = 0; i < num_pairs; i++)
    {

        FeatureLine *first = nullptr;
        FeatureLine *second = nullptr;

        readResult = readPair(fh, &first, &second);

        if( MORPH_FILE_READ_INCORRECT_PAIR == readResult )
        {
            fprintf(stderr, "Invalid pair read\n");

            if(nullptr != first) delete first;
            if(nullptr != second) delete second;

            abort();
        }

        // Add line pair to the FeatureLineManager
        fl_manager->addLinePair(first, second);
    }


    fclose(fh);


}

int MorphFileReader::readPair(FILE *fh, FeatureLine **first, FeatureLine **second)
{
    float firstLine[4];
    float secondLine[4];

    int res;
    char c;

    res = fscanf(fh, "%e,%e,%e,%e[^\n]%c", &firstLine[0], &firstLine[1], &firstLine[2], &firstLine[3], &c);

    // If we matched less than 4 floats, we abort
    if( res < 4 ) return res;


    res = fscanf(fh, "%e,%e,%e,%e[^\n]%c", &secondLine[0], &secondLine[1], &secondLine[2], &secondLine[3], &c);

    // If we matched less than 4 floats, we abort
    if( res < 4 ) return res;


    *first = new FeatureLine{
        firstLine[0],
        firstLine[1],
        firstLine[2],
        firstLine[3]
    };

    *second = new FeatureLine {
        secondLine[0],
        secondLine[1],
        secondLine[2],
        secondLine[3]
    };

    return res;

}

int MorphFileReader::readNumPairs(FILE *fh, int *numPairs)
{
    int num_matched;
    char trailing_character;
    num_matched = fscanf(fh, "%d[^\n]", numPairs);

    // Check if the number was followed by a newline
    trailing_character = fgetc(fh);

    if ( '\n' != trailing_character )
        return MORPH_FILE_READ_NUM_PAIRS_MISSING;

    return MORPH_FILE_READ_SUCCESS;
}



void MorphFileWriter::write(const std::string &path, FeatureLineManager *fl_manager)
{
    int num_pairs = fl_manager->line_pairs.size();

    printf("Num pairs to write: %d\n", num_pairs);

    FILE *fh = fopen(path.c_str(), "w");

    fprintf(fh, "%d\n", num_pairs);

    for( FL_Pair *pair : fl_manager->line_pairs )
    {
        fprintf(fh, "%f,%f,%f,%f\n",
                pair->first->start->x,
                pair->first->start->y,
                pair->first->end->x,
                pair->first->end->y
               );
        fprintf(fh, "%f,%f,%f,%f\n",
                pair->second->start->x,
                pair->second->start->y,
                pair->second->end->x,
                pair->second->end->y
               );
    }

    fclose(fh);



}
