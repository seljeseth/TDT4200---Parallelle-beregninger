#ifndef _FILEREADER_HPP_
#define _FILEREADER_HPP_

#include <string>
#include <memory>

namespace FileReader {
    std::unique_ptr<std::string> readFile(const char* filePath);
}

#endif
