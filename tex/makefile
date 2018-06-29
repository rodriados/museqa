DEPS := $(shell find . -name '*.main.tex')
FILE := $(DEPS:./%.tex=%)

.phony: all clean generate

all: generate

clean:
	@rm -f *.aux *.bbl *.blg *.log *.pdf

generate: $(FILE)

$(FILE): $(DEPS)
	pdflatex $<
	pdflatex $<
	pdflatex $<
	pdflatex $<