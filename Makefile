IBP_report.pdf: IBP_report.tex
	pdflatex IBP_report.tex
	bibtex IBP_report.aux
	pdflatex IBP_report.tex
	pdflatex IBP_report.tex
    
naive: functions.py naive.py
	python functions.py
	python naive.py

usable: functions.py naive.py
	python functions.py
	python usable.py

Cythonized: Cython_setup.py Cython_functions.pyx Cythonized.py
	python Cython_setup.py build_ext --inplace
	python Cythonized.py
    
comparison:
	cd PyIBP; make