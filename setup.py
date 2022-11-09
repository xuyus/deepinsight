import distutils.command.build
import os
import subprocess

import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install
from setuptools import setup, find_packages


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, "deepinsight")

try:
    git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, "VERSION")) as version_file:
    VersionInfo = version_file.readline().strip() + "." + git_version[:8]


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        setuptools.command.build_py.build_py.run(self)


class build(distutils.command.build.build):
    def run(self):
        self.run_command("build_py")


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command("build")
        setuptools.command.develop.develop.run(self)


cmdclass = {
    "build_py": build_py,
    "build": build,
    "develop": develop,
}

setup(
    name="deepinsight",
    version=VersionInfo,
    description="",
    cmdclass=cmdclass,
    packages=find_packages(),
    license="Apache License v2.0",
    author="",
    author_email="",
    url="",
    install_requires=["numpy>=1.14.1", "onnx>=1.4.1", "pandas", "nvtx", "black"],
    scripts=["bin/deepinsight", "bin/deepinsight_py"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
