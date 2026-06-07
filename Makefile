.PHONY: setup test run compare benchmark dqn plot clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/python run.py policy-iteration

compare:
	.venv/bin/python run.py compare

benchmark:
	.venv/bin/python run.py benchmark

dqn:
	.venv/bin/python run.py train-dqn --episodes 150

plot:
	.venv/bin/python run.py plot

leaderboard:
	.venv/bin/python run.py leaderboard

test:
	MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/
