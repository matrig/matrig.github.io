.PHONY: all update-date update-sitemap publications

all: update-date update-sitemap publications

update-date:
	sed -i "s/(last modified: .*)/(last modified: $$(date '+%d %B %Y'))/" index.html

update-sitemap:
	sed -i "s/<lastmod>[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}<\/lastmod>/<lastmod>$$(date '+%Y-%m-%d')<\/lastmod>/g" sitemap.xml

publications:
	$(MAKE) -C publications all
