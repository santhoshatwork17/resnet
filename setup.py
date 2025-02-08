from setuptools import setup, find_packages

setup(
    name="resnet_transfer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "joblib",
        "Pillow"
    ],
    entry_points={
        "console_scripts": [
            "train-resnet=resnet_transfer.train:train_model",
            "predict-resnet=resnet_transfer.predict:predict_image"
        ]
    },
    author="Santhoshkumar",
    author_email="santhoshatwork17@gmail.com",
    description="A simple ResNet50 transfer learning package for image classification.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/santhosh1705kumar/resnet_transfer",
    license="MIT",  # <--- Correct way to specify the license
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
