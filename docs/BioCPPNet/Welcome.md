# Welcome to BioCPPNet

**BioCPPNet** is a bioacoustic source separation project designed to solve the Cocktail Party Problem for non-human vocalizations. This vault serves as the central knowledge base for the project.

## 📂 Navigation

- [[architecture|Architecture Overview]]: Understanding the system design, folder structure, and Python modules.
- [[KANBAN|Project Kanban Board]]: Track tasks, bugs, and feature requests.

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Quick Start (Docker)
Run the application using Docker Compose:
```bash
docker-compose up --build -d
```

### Local Development
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. **Setup Pre-commit Hooks:**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Tools

- **Linting:** We use `ruff` for fast Python linting.
  ```bash
  docker-compose run biocppnet python manage.py lint
  ```
- **Testing:** We use `pytest` for unit testing.
  ```bash
  docker-compose run biocppnet python manage.py test
  ```
- **CLI Management:** The `manage.py` script provides project-specific commands (e.g., `mix_data`, `train`).

## 📝 Notes
- This documentation is built to be viewed in **Obsidian**.
- The `docs/` folder is mapped directly to this vault.
