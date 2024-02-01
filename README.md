# Transfer Learning for CFRP Tape Laying

This project applies transfer learning to improve the efficiency of laying carbon fiber reinforced polymer (CFRP) tapes.

## Getting Started

Below you'll find instructions to set up the project on your local machine for development and testing.

### Prerequisites

Ensure you have the following tools installed:

- Poetry (for dependency management)
- Docker (if you plan to use a development container for GPU support)

### Installation

Set up your S3 access credentials. In your terminal, run:

```bash
export AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
```

Replace <YOUR_AWS_ACCESS_KEY_ID> and <YOUR_AWS_SECRET_ACCESS_KEY> with your S3 credentials.

Use Poetry to install dependencies and set up your environment:

``` bash
poetry install
source .venv/bin/activate
```

Initialize DVC for data version control with S3:
``` bash
dvc init
```

To reproduce the data processing pipeline, execute:
``` bash
dvc repro
```

## Development Container for GPU Support

When you have nvidia-docker installed you can leverage GPU support using a development container by following these steps:

Create .devcontainer/devcontainer.env with S3 Credentials:

``` bash
AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
```

Launch the Development Container and use your container-compatible IDE (e.g., Visual Studio Code) to open the project in a development container, ensuring it's configured for GPU support.