import setuptools

setuptools.setup(
    name="pdpf",
    version="0.0.1",
    author="mirucaaura",
    description="python implementation of Primal-Dual Path-Following for linear programming",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"}
)
