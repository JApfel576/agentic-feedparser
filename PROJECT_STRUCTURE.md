```
.
├── Dockerfile
├── LICENSE
├── PROJECT_STRUCTURE.md
├── README.Docker.md
├── README.md
├── compose.yaml
├── packages
│   ├── base_agents
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   └── src
│   │       ├── base_agents
│   │       │   ├── __init__.py
│   │       │   └── base_agents.py
│   │       └── base_agents.egg-info
│   │           ├── PKG-INFO
│   │           ├── SOURCES.txt
│   │           ├── dependency_links.txt
│   │           ├── requires.txt
│   │           └── top_level.txt
│   ├── feed_agents
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   └── src
│   │       └── feed_agents
│   │           ├── __init__.py
│   │           └── feed_agents.py
│   └── feedpoller
│       ├── pyproject.toml
│       └── src
│           ├── feedpoller
│           │   ├── __init__.py
│           │   └── feedpoller.py
│           └── feedpoller.egg-info
│               ├── PKG-INFO
│               ├── SOURCES.txt
│               ├── dependency_links.txt
│               ├── requires.txt
│               └── top_level.txt
├── pyproject.toml
├── services
│   └── fastapi_app
│       ├── pyproject.toml
│       └── src
│           └── fastapi_app
│               └── main.py
├── uv.lock
└── var
    └── data
        ├── bbc_com
        │   ├── 20260515_183820.json
        │   └── state.json
        └── reuters_com
            ├── 20260515_183846.json
            └── state.json

21 directories, 35 files
```
