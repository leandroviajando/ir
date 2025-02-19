SHELL     := /usr/bin/env bash
MAKEFLAGS += --silent

default: compile

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies
	brew install pandoc

.PHONY: compile
compile: ## Compile the documents
	pandoc -V geometry:margin=1in -V fontfamily=helvet --pdf-engine=xelatex -o Information\ Retrieval\ and\ Search.pdf README.md
