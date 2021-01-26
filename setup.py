from setuptools import setup, find_packages
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

setup(
    name="fingerveinapp",
    packages=find_packages(),
    version="1.0.0",
    description="vs80 finger vein tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/1wickes/FingerVeinApp",
    author="",
    author_email="",
    license="MIT",
    classifiers=classifiers,
    keywords="collections",
    install_requires=[""],
)