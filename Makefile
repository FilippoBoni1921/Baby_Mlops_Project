########################################################
# Utilities
########################################################
# Force Make to be silent
ifndef VERBOSE
.SILENT:
endif

# Get the OS name
UNAME := $(shell uname)

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

# Include Environment Variables
-include ../.env

default: help
.PHONY: help
help:
	@grep -hE '^[ a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

########################################################
# Pipelines
########################################################
train_pipeline: setup_ssh copy training copy_from_vm cleanup_ssh

########################################################
# Running
########################################################

copy:
	@$(SSH_SETUP); \
	echo "Copying files from $(LOCAL_PATH) to $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)"; \
	tar -X $(SSH_IGNORE) --no-same-owner -czf ./../workspace.tar.gz -C "$(LOCAL_PATH)" . && \
	echo "Ensuring remote directory exists..."; \
	ssh -p $(SSH_PORT) -i $(SSH_KEY) $(REMOTE_USER)@$(REMOTE_HOST) "mkdir -p $(REMOTE_PATH)" && \
	echo "Archive created, starting SCP transfer..."; \
	scp -P $(SSH_PORT) -i $(SSH_KEY) ./../workspace.tar.gz $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH) && \
	echo "SCP transfer completed, removing local archive"; \
	rm ./../workspace.tar.gz && \
	echo "Extracting remote archive..."; \
	ssh -p $(SSH_PORT) -i $(SSH_KEY) $(REMOTE_USER)@$(REMOTE_HOST) "tar --no-same-owner -xzf $(REMOTE_PATH)workspace.tar.gz -C $(REMOTE_PATH) && rm $(REMOTE_PATH)workspace.tar.gz" && \
	echo "Remote extraction completed"
.PHONY: copy

ssh:
	@echo "Connecting to $(REMOTE_USER)@$(REMOTE_HOST)"
	ssh -i $(SSH_KEY) -p $(SSH_PORT) $(REMOTE_USER)@$(REMOTE_HOST)
.PHONY: ssh

create_dataset:
	@echo "Creating Dataset"
	PYTHONPATH=$(LOCAL_PATH) python src/finetuning/data_loader.py
.PHONY: create_dataset

training:
	@$(SSH_SETUP); \
	echo "Performing Training"; \
	ssh -i $(SSH_KEY) -p $(SSH_PORT) $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_PATH) && PYTHONPATH=$(REMOTE_PATH) python src/finetuning/training.py'
.PHONY: training

copy_from_vm:
	@$(SSH_SETUP); \
	echo "Copying the finetuning results locally"; \
	scp -P $(SSH_PORT) -r -i $(SSH_KEY) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_MODELS_PATH) "$(LOCAL_MODELS_PATH)"
	scp -P $(SSH_PORT) -r -i $(SSH_KEY) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOVE_RESULTS_PATH) "$(LOCAL_RESULTS_PATH)"
.PHONY: copy_from_vm