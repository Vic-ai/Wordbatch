IMAGE_NAME=wordbatch
CONTAINER_NAME=wordbatch_dev

# macOS ARM (Apple Silicon) build flags - uncomment if not set in your shell environment
# export CFLAGS := -I/opt/homebrew/include -I/opt/homebrew/opt/libomp/include
# export LDFLAGS := -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib
# export CXXFLAGS := $(CFLAGS)

.PHONY: wheel clean build run-dev attach stop test

wheel: ## Build wheel for current platform
	@echo "Building wheel... (macOS ARM users: ensure CFLAGS/LDFLAGS are set in your environment or uncomment them above)"
	pip wheel --no-build-isolation --no-deps -w dist/ .

wheel-linux: ## Build wheel for Linux x86_64 (via Docker)
	docker run --rm --platform linux/amd64 -v $(PWD):/wordbatch -w /wordbatch python:3.11 bash -c \
		"pip install 'Cython>=0.29.20' 'numpy>=1.23.2,<2.0' wheel && pip wheel --no-build-isolation --no-deps -w dist/ ."

upload: ## Upload wheels to private PyPI. Usage: make upload VERSION=2.1.0
	@test -f .pypi.env || (echo "Error: .pypi.env not found. Create it with PYPI_URL, PYPI_USER, PYPI_PASSWORD" && exit 1)
	@test -n "$(VERSION)" || (echo "Error: VERSION is required. Usage: make upload VERSION=2.1.0" && exit 1)
	pip install twine --quiet
	@set -a && . ./.pypi.env && set +a && \
		twine upload --repository-url $$PYPI_URL -u $$PYPI_USER -p $$PYPI_PASSWORD dist/wordbatch-$(VERSION)-*.whl

clean: ## Clean build artifacts
	rm -rf build/ *.egg-info/
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

run-dev: ## Run container for development
	docker run \
		-it \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/wordbatch $(IMAGE_NAME) bash

attach: ## Run a bash in a running container
	docker start $(CONTAINER_NAME) && docker attach $(CONTAINER_NAME)

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)

test:
	pytest
	docker start $(CONTAINER_NAME)
	docker exec -it $(CONTAINER_NAME)  env | grep PATH
	docker exec -it $(CONTAINER_NAME)  which python
	docker exec -it $(CONTAINER_NAME) python -c '\
	import wordbatch;\
	from wordbatch import models;\
	print(wordbatch.__version__)'
	#docker exec -it $(CONTAINER_NAME) python -c 'print("hello")'
	#docker exec -it $(CONTAINER_NAME) echo "Hello from container!"