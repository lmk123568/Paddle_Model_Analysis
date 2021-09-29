from setuptools import find_packages, setup


VERSION = "0.3.1"
REQUIREMENTS = [
    "rich",
    "opencv-python",
]


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":

    setup(
        name="ppma",
        version=VERSION,
        description="PaddlePaddle Model Analysis.",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="Mike",
        author_email="lmk123568@qq.com",
        python_requires=">=3.6.0",
        url="https://github.com/lmk123568/Paddle_Model_Analysis",
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
    )
