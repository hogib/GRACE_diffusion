# Phase 1 lab report.
#   make figures   regenerate figures from runs/eval/*.csv (copied from the server)
#   make           compile the PDF
all: phase1_report.pdf

phase1_report.pdf: phase1_report.tex $(wildcard figures/*.pdf)
	tectonic -X compile phase1_report.tex

figures:
	cd ../.. && uv run --with pandas python reports/phase1/make_figures.py

.PHONY: all figures
