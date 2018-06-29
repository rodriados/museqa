FILES := $(shell find . -name '*.main.tex' | awk -F. '{$NF=""; print $0}' | tr ' ' '.' | rev | cut -c 2- | rev)

.phony: all clean generate

all: generate

clean:
	@rm *.aux *.bbl *.blg *.log *.pdf

generate: $(FILES)
	pdflatex $<
	bibtex   $<
	pdflatex $<
	pdflatex $<
	pdflatex $<
