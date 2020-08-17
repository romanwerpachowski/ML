# SCons build script for ML project on Linux.
# (C) 2020 Roman Werpachowski.

import os
import SCons


# Get the mode flag from the command line.
build_mode = ARGUMENTS.get('mode', 'debug')
Export('build_mode')

# Only 'debug' or 'release' allowed.
if not (build_mode in ['debug', 'release']):
    print("Error: expected 'debug' or 'release' for 'build_mode' parameter, found: " + build_mode)
    Exit(1)
   
print('**** Compiling in %s mode ****' % build_mode)

# Extra compile flags for debug mode.
debugcflags = ['-g']
# Extra compile flags for release mode.
releasecflags = ['-O2', '-flto', '-DNDEBUG']

# Paths for Ubuntu LTS.
EIGEN_PATH = '/usr/include/eigen3/'
system_include_paths = [('-isystem' + path) for path in [EIGEN_PATH]] # no space after -isystem!
compilation_options = ['-fno-strict-overflow', '-fdiagnostics-color', '-march=native']
enabled_warnings = ['-Wall', '-Werror', '-Wfatal-errors', '-Wpedantic', '-Wformat', '-Wextra', '-Wconversion']
c_flags = system_include_paths + compilation_options + enabled_warnings
linkflags = []
arch_switch = '-m64'
c_flags.append(arch_switch)
linkflags.append(arch_switch)
flags = ["-std=c++17"] + c_flags
BUILD_DIR = 'build/%s' % build_mode.capitalize()
if build_mode == 'debug':
    flags += debugcflags
else:
    flags += releasecflags
Export('BUILD_DIR')

cpp_path = ['#']
ar = 'ar'
ranlib = 'ranlib'
env = Environment(CPPPATH=cpp_path,
                  CXXFLAGS=flags,
                  CFLAGS=c_flags,
                  LINKFLAGS=linkflags,
                  AR=ar,
                  RANLIB=ranlib,
                  ENV={'PATH' : os.environ['PATH'],
                       'TERM' : os.environ['TERM'],
                       'HOME' : os.environ['HOME']}
)
Export('env')

# Linked libraries.
OTHER_LIBS = []
Export('OTHER_LIBS')

def call(subdir, name='SConscript'):
    return SConscript(os.path.join(subdir, name), variant_dir = os.path.join(subdir, BUILD_DIR), duplicate = 0)

# Build libraries.
ML, MLObjs = call('ML')
Export('ML')
Export('MLObjs')
call('Demo')
call('PythonBindings')

if build_mode == 'debug':
    top_dir = Dir('#').abspath   
    env.Append(CXXFLAGS=['-isystem' + os.path.join(top_dir, 'googletest/include')])
    gtest = call('googletest', 'SConscript-gtest')
    Export('gtest')
    GTEST_LIBS = ['pthread']
    Export('GTEST_LIBS')
    call('Tests')    
else:
    call('Benchmarks')