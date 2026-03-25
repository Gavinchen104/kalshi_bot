from setuptools import find_packages, setup


setup(
    name="kalshi-bot",
    version="0.1.0",
    description="Starter scaffold for an automated Kalshi trading bot.",
    package_dir={"": "."},
    packages=find_packages(where="."),
    install_requires=[
        "httpx>=0.27.0",
        "cryptography>=43.0.0",
        "websockets>=12.0",
        "pydantic>=2.8.0",
        "pydantic-settings>=2.4.0",
        "PyYAML>=6.0.1",
        "structlog>=24.2.0",
        "streamlit>=1.36.0",
        "streamlit-autorefresh>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.8",
            "ruff>=0.5.0",
        ]
    },
)

