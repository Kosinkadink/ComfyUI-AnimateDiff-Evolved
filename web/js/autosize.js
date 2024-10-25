import { app } from '../../../scripts/app.js'

function addResizeHook(node, padding, useOldMin=false) {
    let origOnCreated = node.onNodeCreated
    node.onNodeCreated = function() {
        let r = origOnCreated?.apply(this, arguments)
        let size = this.computeSize();
        size[0] += padding || 0;
        if (useOldMin) {
            //equal to LiteGraph.NODE_WIDTH*1.5*1.5
            size[0] = Math.max(size[0], 315)
        }
        this.setSize(size);
        return r
    }
}

app.registerExtension({
    name: "AnimateDiffEvolved.autosize",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        //since python_module is based off folder path,
        //it could be changed by users and should only be used as fallback
        if (nodeData?.name?.startsWith("ADE_")
            || nodeData.python_module == 'custom_nodes.ComfyUI-AnimateDiff-Evolved') {
            if (nodeData?.input?.hidden?.autosize) {
                addResizeHook(nodeType.prototype, nodeData.input.hidden.autosize[1]?.padding)
            } else if (!nodeData?.input?.optional?.autosize) {
                addResizeHook(nodeType.prototype, 0, true)
            }
        }
    },
    async getCustomWidgets() {
        return {
            ADEAUTOSIZE(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "ADE.AUTOSIZE",
                    value : "",
                    options : {"serialize": false},
                    computeSize : function(width) {
                        return [0, -4];
                    }
                }
                if (!node.widgets) {
                    node.widgets = []
                }
                node.widgets.push(w)
                addResizeHook(node, inputData[1].padding);
                return w;
            }
        }
    }
});
