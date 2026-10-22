{%- assign ref = include.ref %}
{%- include paper_url.html ref=ref -%}
- [{{ ref.title }}]({{ url | absolute_url }}). {{ ref.authors }}. {{ ref.publication }}, {{ ref.issue }} {{ ref.year }}.
{%- if ref.pdf %} [PDF]({{ ref.pdf | absolute_url }}){% endif %}
{%- if include.show_note and ref.note %}
  {{ ref.note }}
{%- endif -%}
