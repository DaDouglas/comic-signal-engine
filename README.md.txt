Comic ML Signal

A prototype ML system that clusters comic book issues based on
metadata features and returns a proxy collectibility signal.

Components:
- build_artifacts.py → training pipeline
- scoring.py → scoring interface
- app.py → Flask web app

Run:
1. python build_artifacts.py
2. python app.py
3. open http://127.0.0.1:5000