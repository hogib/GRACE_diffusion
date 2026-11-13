# Build the GRACE + diffusion literature survey.
#   make          merge notes/*.bib, regenerate TO_FETCH.md, compile the PDF
#   make check    list cited keys that are not FULLTEXT_READ (should print only imonet)
#   make clean    remove build products
PY := python3 -I

all: grace_diffusion_survey.pdf

build/papers.tsv build/references.bib: $(wildcard notes/*.bib notes/*.md) tools/collect.py
	$(PY) tools/collect.py notes build

references.bib: build/references.bib
	command cp -f $< $@

TO_FETCH.md: build/papers.tsv tools/fetchlist.py
	$(PY) tools/fetchlist.py notes build $@

grace_diffusion_survey.pdf: grace_diffusion_survey.tex $(wildcard sections/*.tex) references.bib TO_FETCH.md
	tectonic -X compile grace_diffusion_survey.tex

check: build/papers.tsv
	@grep -oh '\\cite{[^}]*}' sections/*.tex | sed 's/\\cite{//;s/}//' | tr ',' '\n' | sort -u > build/cited.txt
	@awk -F'\t' 'NR==FNR{st[$$1]=$$3; next} {s=st[$$1]; if(s!="FULLTEXT_READ") print $$1, (s==""?"MISSING":s)}' build/papers.tsv build/cited.txt

clean:
	rm -rf build grace_diffusion_survey.log

.PHONY: all check clean
