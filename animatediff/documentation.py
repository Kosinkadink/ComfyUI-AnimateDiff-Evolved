from typing import Union

from .logger import logger

def image(src):
    return f'<img src={src} style="width: 0px; min-width: 100%">'
def video(src):
    return f'<video src={src} autoplay muted loop controls controlslist="nodownload noremoteplayback noplaybackrate" style="width: 0px; min-width: 100%" class="VHS_loopedvideo">'
def short_desc(desc):
    return  f'<div id=VHS_shortdesc style="font-size: .8em">{desc}</div>'

def coll(text: str):
    return f"{text}_collapsed"

descriptions = {
}

sizes = ['1.4','1.2','1']
def as_html(entry, depth=0):
    if isinstance(entry, dict):
        size = 0.8 if depth < 2 else 1
        html = ''
        for k in entry:
            if k == "collapsed":
                continue
            collapse_single = k.endswith("_collapsed")
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            collapse_flag = ' VHS_precollapse' if entry.get("collapsed", False) or collapse_single else ''
            html += f'<div vhs_title=\"{name}\" style=\"display: flex; font-size: {size}em\" class=\"VHS_collapse{collapse_flag}\"><div style=\"color: #AAA; height: 1.5em;\">[<span style=\"font-family: monospace\">-</span>]</div><div style=\"width: 100%\">{name}: {as_html(entry[k], depth=depth+1)}</div></div>'
        return html
    if isinstance(entry, list):
        html = ''
        for i in entry:
            html += f'<div>{as_html(i, depth=depth)}</div>'
        return html
    return str(entry)


def register_description(node_id: str, desc: Union[list, dict]):
    descriptions[node_id] = desc


def format_descriptions(nodes):
    for k in descriptions:
        if k.endswith("_collapsed"):
            k = k[:-len("_collapsed")]
        nodes[k].DESCRIPTION = as_html(descriptions[k])
    # undocumented_nodes = []
    # for k in nodes:
    #     if not hasattr(nodes[k], "DESCRIPTION"):
    #         undocumented_nodes.append(k)
    # if len(undocumented_nodes) > 0:
    #     logger.info(f"Undocumented nodes: {undocumented_nodes}")


class DocHelper:
    def __init__(self):
        self.actual_dict = {}
    
    def add(self, add_dict):
        self.actual_dict.update(add_dict)
        return self

    def get(self):
        return self.actual_dict
    
    @staticmethod
    def combine(*args):
        docs = DocHelper()
        for doc in args:
            docs.add(doc)
        return docs.get()
