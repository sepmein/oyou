clean:
	rm -r dist
	rm -r oyou.egg-info

rebuild:
	python setup.py sdist
	python setup.py bdist_wheel

publish:
	twine upload dist/*

test:
    python -m unittest test/*