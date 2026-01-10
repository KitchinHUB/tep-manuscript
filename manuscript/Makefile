.PHONY: pdf clean help

# Default target
.DEFAULT_GOAL := help

# Main document
MAIN := main

##@ Help
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\\nUsage:\\n  make \\033[36m<target>\\033[0m\\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \\033[36m%-15s\\033[0m %s\\n", $$1, $$2 } /^##@/ { printf "\\n\\033[1m%s\\033[0m\\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build
pdf: $(MAIN).pdf ## Build manuscript PDF

$(MAIN).pdf: $(MAIN).tex body.tex references.bib
	@echo "Building manuscript PDF..."
	-@pdflatex -interaction=nonstopmode $(MAIN).tex
	-@bibtex $(MAIN)
	-@pdflatex -interaction=nonstopmode $(MAIN).tex
	-@pdflatex -interaction=nonstopmode $(MAIN).tex
	@echo "✓ PDF built: $(MAIN).pdf"

##@ Utilities
clean: ## Remove build artifacts
	@echo "Cleaning build files..."
	@rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	@echo "✓ Build files cleaned"

clean-all: clean ## Remove build artifacts and PDF
	@echo "Removing PDF..."
	@rm -f $(MAIN).pdf
	@echo "✓ All build files removed"

watch: ## Continuously rebuild on changes (requires fswatch)
	@echo "Watching for changes..."
	@fswatch -o $(MAIN).tex body.tex references.bib | xargs -n1 -I{} make pdf
