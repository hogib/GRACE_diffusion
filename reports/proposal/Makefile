# Proposal.
#   make figures   regenerate the figure
#   make           compile the PDF
all: proposal.pdf

proposal.pdf: proposal.tex $(wildcard figures/*.pdf)
	tectonic -X compile proposal.tex

figures:
	cd ../.. && uv run python reports/proposal/make_figures.py

.PHONY: all figures
