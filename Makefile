VENV := venv
BENCHMARKS = $(basename $(notdir $(wildcard benchmarks/benchmark_*.py)))
TESTS = $(basename $(notdir $(wildcard tests/test_*.py)))

$(VENV)/bin/activate: requirements.txt
	python -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt


.PHONY: venv
venv: $(VENV)/bin/activate


.PHONY: tests
tests: venv
	@echo "Testing everything"
	./$(VENV)/bin/python -m pytest

$(TESTS): venv
	@echo "Testing $@"
	./$(VENV)/bin/python -m pytest -k $@


.PHONY: benchmarks
benchmarks: $(BENCHMARKS)

$(BENCHMARKS): venv
	@echo "Benchmarking $@"
	./$(VENV)/bin/python benchmarks/$@.py


.PHONY: clean
clean:
	rm -rf $(VENV) .pytest_cache
