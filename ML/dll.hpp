#pragma once
/* (C) 2020 Roman Werpachowski. */

/** @file dll.hpp
Preprocessor declarations used to export functions and classes in the shared library.
*/

#ifdef MLPP_IS_STATIC
#define DLL_DECLSPEC 
#else
#ifdef _MSC_VER
#ifdef _EXPORTING
#define DLL_DECLSPEC    __declspec(dllexport)
#else
#define DLL_DECLSPEC    __declspec(dllimport)
#endif // _EXPORTING
#else
#define DLL_DECLSPEC 
#endif // _MSV_VER
#endif // MLPP_IS_STATIC