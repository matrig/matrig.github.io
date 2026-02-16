.PHONY: all update-date publications

all: update-date publications

update-date:
	sed -i "s/(last modified: .*)/(last modified: $$(date '+%d %B %Y'))/" index.html
	sed -i "/matrig\.net\/<\/loc>/{n;s|<lastmod>[^<]*</lastmod>|<lastmod>$$(date '+%Y-%m-%d')</lastmod>|}" sitemap.xml

publications:
	$(MAKE) -C publications all
