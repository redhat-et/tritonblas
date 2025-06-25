BENCHMARKS = $(basename $(notdir $(wildcard benchmarks/benchmark_*.py)))
TESTS = $(basename $(notdir $(wildcard tests/test_*.py)))


.PHONY: tests
tests:
	@echo "Running all tests"
	python -m pytest

$(TESTS):
	@echo "Running $@"
	python -m pytest -k $@


.PHONY: benchmarks
benchmarks: $(BENCHMARKS)

$(BENCHMARKS):
	@echo "Running $@"
	python benchmarks/$@.py


.PHONY: clean
clean:
	rm -rf .pytest_cache
