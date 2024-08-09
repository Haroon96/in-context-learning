import json
import os

def update_macro(name, value):

    if not os.path.exists('output'):
        os.makedirs('output')

    # load macros from disk
    try:
        with open('output/macros.json') as f:
            js = json.load(f)
    except FileNotFoundError as e:
        js = {}

    # update macro value
    js[name] = value

    # write macros back to disk
    with open('output/macros.json', 'w') as f:
        json.dump(js, f, indent=2)

    # write tex file
    with open('output/macros.tex', 'w') as t:
        t.write('%% AUTO GENERATED FILE\n')
        for k, v in js.items():
            v = str(v).replace('%', '\%')
            t.write('\\newcommand{\\%s}{%s}\n' % (k, v))
