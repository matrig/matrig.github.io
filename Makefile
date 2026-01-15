.PHONY: all update-date publications

all: update-date publications

update-date:
	sed -i "s/(last modified: .*)/(last modified: $$(date '+%d %B %Y'))/" index.html

publications:
	$(MAKE) -C publications all
