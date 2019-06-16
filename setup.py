import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="TimeSeriesVisualization",
        version="1.0",
        author="Ameya Daigavane",
        author_email="ameya.d.98@gmail.com",
        description="Time-series Visualization with the Matrix Profile and Multidimensional Scaling.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ameya98/TimeSeriesVisualization",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )

