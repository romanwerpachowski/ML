#pragma once

#ifdef _EXPORTING
#define DLL_DECLSPEC    __declspec(dllexport)
#else
#define DLL_DECLSPEC    __declspec(dllimport)
#endif // _EXPORTING