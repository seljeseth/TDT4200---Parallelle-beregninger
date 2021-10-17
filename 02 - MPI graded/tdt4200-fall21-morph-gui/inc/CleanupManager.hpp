#ifndef _CLEANUPMANAGER_HPP_
#define _CLEANUPMANAGER_HPP_

#include <vector>

class CleanupManager
{
public:

    using vfp = void (*)();
    static std::vector<vfp> function_pointers;

    static void init();

    static void registerHandler(vfp handler);

    static void doCleanup();

private:
    CleanupManager() = delete;


};

#endif
