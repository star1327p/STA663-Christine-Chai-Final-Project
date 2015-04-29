IBP_report.pdf: IBP_report.tex
	pdflatex IBP_report.tex
	bibtex IBP_report.aux
	pdflatex IBP_report.tex
	pdflatex IBP_report.tex
    
wrong: 
	cd Version0_Wrong; python functions.py
	cd Version0_Wrong; python naive_wrong.py
    
naive:
	cd Version1_Naive_Code; python functions.py
	cd Version1_Naive_Code; python naive.py
    
reverse: 
	cd Reverse; python functions.py
	cd Reverse; python reverse.py

testing: 
	cd Unit_Testing; python unit_function_tests.py
	cd Unit_Testing; python unit_likelihood_tests.py

usable:
	cd Version2_Usable_Code; python functions.py
	cd Version2_Usable_Code; python usable.py

Cythonized:
	cd Version3_Cythonized_Code; python Cython_setup.py build_ext --inplace
	cd Version3_Cythonized_Code; python Cythonized.py
    
Using_jit:
	cd Version4_jit_Code; python jit_functions.py
	cd Version4_jit_Code; python jit_IBPcode.py    
    
comparison:
	cd PyIBP; make