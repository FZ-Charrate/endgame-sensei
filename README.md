chesscoach-rl/
├── src/
│   └── chess_coach/
│       ├── engine/          # board simulator, move validation (wraps python-chess)
│       ├── rl/               # agent, training loop, reward logic
│       ├── classifier/       # endgame pattern detection rules
│       ├── rag/               # chunking, embedding, retrieval, LLM calls
│       ├── db/                 # Postgres models/queries (games, moves, vectors)
│       └── api/                # FastAPI app, routes
├── data/
│   ├── raw/                  # downloaded/scraped source data (lichess games, wikipedia text)
│   └── processed/            # cleaned/chunked data ready for use (ETL output)
├── notebooks/                # exploration, RL training experiments, RAG quality checks
├── models/                   # saved trained RL agent weights
├── tests/                     # unit tests for engine, classifier, rag logic
├── scripts/                   # one-off runnable scripts (e.g. run_etl.py, train_agent.py)
├── config/                    # config files (.env.example, settings.yaml)
├── docker-compose.yml         # Postgres + app, so it's reproducible
├── requirements.txt / pyproject.toml
└── README.md