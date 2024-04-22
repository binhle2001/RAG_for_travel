## Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
## Run project
python vector_database.py
python socket_server.py
python -m http.server 8081
## using sockerUrl = localhost:8765
## web demo = localhost:8081
