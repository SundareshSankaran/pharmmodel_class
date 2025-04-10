# PHARMMODEL

An extensible Python class for analytics and data science.

This repository has been developed for the purpose of facilitating a PharmaSUG 2025 paper on "Building Extensible Python Classes: It's Easier Than You Think!" Refer [here](https://pharmasug.org/conferences/pharmasug-2025-us/paper-presentations/#OS-364) for paper details.

## Abstract

Data Analytics in the life sciences industry requires iteration and access to a wide range of methods and techniques for trusted, stable and robust results. The Python open-source ecosystem provides a rich array of such methods through a number of packages and modules that promote rapid and flexible experimentation. However, such variety also has its downsides such as package dependencies, strict version support and a clutter of similar packages. Data scientists and programmers require a common framework which packages many capabilities in a seamless manner. In this session, we provide a design and example of a Python class which encapsulates established packages and pipelines for data management and outcome predictions, making them available from a single instance. These methods cover a range of operations across the analytics life cycle, and are extensible to include new methods and classes. Every instance can be associated with the source dataset and analytical artifacts created in-process, and can be encapsulated into a single package for porting to other environments, thus enabling easy promotion of analysis. We also make available source code and examples for working with this class, and explain how this can be customised for your organisation’s specific needs. This session provides the audience valuable tools and knowledge on how to organise their Python code in a structured framework, and gain efficiency and productivity benefits.

## Installation

1. Clone this repo to your workstation

```
git clone https://github.com/SundareshSankaran/pharmmodel_class.git
```

2. Optional : You may like to construct a virtual environment (instructions provided in [build.sh](./build/build.sh)) to install  package

3. Install the packages listed under [requirements.txt](./build/requirements.txt) in the [build](./build/) folder

```
pip install -r ./build/requirements.txt
```

4. Install the pharmmod package, containing the pharmmod class. Note that this is a local package.  Instructions also provided in build.sh

```
pip install  -e .
```

## Contact
- [Sundaresh Sankaran](mailto:sundaresh.sankaran@sas.com)
- [Samiul Haque](mailto:samiul.haque@sas.com)
