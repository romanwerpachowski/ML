#pragma once

/** @file dll.hpp
Preprocessor declarations used to export functions and classes in the shared library.
*/

#ifdef _MSC_VER
#ifdef _EXPORTING
#define DLL_DECLSPEC    __declspec(dllexport)
#else
#define DLL_DECLSPEC    __declspec(dllimport)
#endif // _EXPORTING
#else
#define DLL_DECLSPEC 
#endif // _MSV_VER