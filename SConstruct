# SCons build script for ML project on Linux.
# (C) 2020 Roman Werpachowski.

import os
import SCons


# Get the mode flag from the command line.
mymode = ARGUMENTS.get('mode', 'debug')
Export('mymode')

# Only 'debug' or 'release' allowed.
if not (mymode in ['debug', 'release']):
    print("Error: expected 'debug' or 'release' for 'mymode' parameter, found: " + mymode)
    Exit(1)
   
print('**** Compiling in %s mode ****' % mymode)

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
if mymode == 'debug':
    BUILD_DIR = 'scons_build/Debug'
    flags += debugcflags
else:
    BUILD_DIR = 'scons_build/Release'
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
ML = call('ML')
Export('ML')

if mymode == 'debug':
    top_dir = Dir('#').abspath   
    env.Append(CXXFLAGS=['-isystem' + os.path.join(top_dir, 'googletest/include')])
    gtest = call('googletest', 'SConscript-gtest')
    Export('gtest')
    GTEST_LIBS = ['pthread']
    Export('GTEST_LIBS')
    call('Tests')
