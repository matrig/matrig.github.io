.PHONY: all
all: ../articles.html ../conferences.html ../abstracts.html ../patents.html

../articles.html : articles.yaml template_articles.html
	python template_renderer.py \
		 template_articles.html articles.yaml ../articles.html
	sed -i "/articles.html/{n;s|<lastmod>[^<]*</lastmod>|<lastmod>$$(date '+%Y-%m-%d')</lastmod>|}" ../sitemap.xml

../conferences.html : conferences.yaml template_conferences.html
	python template_renderer.py \
		template_conferences.html conferences.yaml ../conferences.html
	sed -i "/conferences.html/{n;s|<lastmod>[^<]*</lastmod>|<lastmod>$$(date '+%Y-%m-%d')</lastmod>|}" ../sitemap.xml

../abstracts.html : abstracts.yaml template_abstracts.html
	python template_renderer.py \
		template_abstracts.html abstracts.yaml ../abstracts.html
	sed -i "/abstracts.html/{n;s|<lastmod>[^<]*</lastmod>|<lastmod>$$(date '+%Y-%m-%d')</lastmod>|}" ../sitemap.xml

../patents.html : patents.yaml template_patents.html
	python template_renderer.py \
		template_patents.html patents.yaml ../patents.html
	sed -i "/patents.html/{n;s|<lastmod>[^<]*</lastmod>|<lastmod>$$(date '+%Y-%m-%d')</lastmod>|}" ../sitemap.xml
