black:
	@black --version
	@ls -d assignment* | while read -r src; do black "$$src"; done

black_check:
	@black --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do black --check "$$src"; done

isort:
	@isort --version
	@isort --recursive .

isort_check:
	@isort --version
	@isort --recursive --check-only

flake8:
	@flake8 --version
	@ls -d assignment* | while read -r src; do flake8 "$$src"; done

clean_notebooks:
    # This finds Ipython jupyter notebooks in the code
    # base and cleans only its output results. This
    # results in
	@jupyter nbconvert --version
	@find . -maxdepth 3 -name '*.ipynb'\
		| while read -r src; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$$src"; done

pylint:
	@pylint --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do pylint -rn "$$src"; done

all:black flake8 clean_notebooks
ci:black_check flake8
