#include <CleanupManager.hpp>
#include <cstdlib>

std::vector<CleanupManager::vfp> CleanupManager::function_pointers{};

void CleanupManager::registerHandler(CleanupManager::vfp handler)
{
    function_pointers.emplace_back(handler);
}

void CleanupManager::doCleanup()
{
    for (vfp handler : function_pointers)
    {
        handler();
    }
}

void CleanupManager::init()
{
    atexit(CleanupManager::doCleanup);
}
