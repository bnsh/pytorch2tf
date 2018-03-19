PYTHON=$(wildcard *.py */*.py)
PYLINT=$(join $(dir $(PYTHON)), $(addprefix ., $(notdir $(PYTHON:py=pylint))))

all: pylint

clean:
	/bin/rm -f $(PYLINT)

pylint: $(PYLINT)

.%.pylint: %.py
	/usr/local/bin/pylint -r n $(^)
	/usr/bin/touch $(@)
