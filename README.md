## Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
## Run project
python vector_database.py
python socket_server.py
## using sockerUrl = localhost:8765
