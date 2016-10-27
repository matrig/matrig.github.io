#!/bin/bash
#@date Wed Oct 26 23:18:12 EDT 2016
#@author Mattia Rigotti
line_parser.py -i articles.txt -t template_articles
mv output articles.html

line_parser.py -i abstracts.txt -t template_abstracts
mv output abstracts.html
