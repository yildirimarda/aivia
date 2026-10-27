# Developer entry points. `make check` is the aggregate gate.
.PHONY: lint type test check docs install-smoke

lint:
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest -q

# Keep the gate deterministic even when make is invoked with -j.
check: lint type test

.NOTPARALLEL: check

docs:
	@test -f docs/DESIGN.md
	@test -f docs/decisions.md
	@echo "Documentation: docs/DESIGN.md, docs/decisions.md"

install-smoke:
	@if [ -x scripts/install_smoke.sh ]; then \
		scripts/install_smoke.sh; \
	else \
		echo "scripts/install_smoke.sh is not present yet; running the packaging smoke"; \
		uv run pytest -q tests/test_packaging.py -k "installed_console_script or module_entry_point"; \
	fi
