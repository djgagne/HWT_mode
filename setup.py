from setuptools import setup

setup(name="hwtmode",
      version="0.1",
      description="Analyze storm mode with machine learning.",
      author="David John Gagne, David Ahijevych",
      author_email="dgagne@ucar.edu",
      license="MIT",
      url="https://github.com/NCAR/HWT_mode",
      packages=["hwtmode"],
      install_requires=["numpy",
                        "xarray",
                        "pandas"])