UV := .tools/uv

.PHONY: bootstrap sync test lint data reproduce validate report app smoke

bootstrap:
	./scripts/bootstrap.sh

sync:
	$(UV) sync --extra dev

test:
	$(UV) run pytest -m "not network and not slow"

lint:
	$(UV) run ruff check src tests app

data:
	$(UV) run index-timing data sync --study all

reproduce:
	$(UV) run index-timing reproduce --study all

validate:
	$(UV) run index-timing validate --study all

report:
	$(UV) run index-timing report build

app:
	$(UV) run index-timing app

smoke:
	$(UV) run index-timing demo

