import sys
import yaml
from jinja2 import Environment, FileSystemLoader

def render(template_file, data_file, output_file):
    # Load YAML data
    with open(data_file, 'r') as file:
        data = yaml.safe_load(file)

    # Sort filtered articles by year in descending order
    if hasattr(data[0], 'year'):
        data.sort(key=lambda x: x['year'], reverse=True)

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_file)

    # Render the template with the filtered and sorted YAML data
    rendered_html = template.render(items=data)

    # Write the rendered HTML to a file
    with open(output_file, 'w') as file:
        file.write(rendered_html)

    print("HTML file has been rendered and saved as", output_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python renderer.py <template_file> <data_file> <output_file>")
        sys.exit(1)

    template_file = sys.argv[1]
    data_file = sys.argv[2]
    output_file = sys.argv[3]

    render(template_file, data_file, output_file)
