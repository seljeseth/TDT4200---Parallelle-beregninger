#include <FileReader.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using std::string;

std::unique_ptr<std::string> FileReader::readFile(const char* filePath)
{

    std::ifstream file;

    std::ostringstream sstr;

    std::unique_ptr<std::string> contents = std::make_unique<std::string>();

    file.open(filePath);

    if(file.is_open() )
    {
        sstr << file.rdbuf();
    }

    *contents = sstr.str();


    return contents;



}
