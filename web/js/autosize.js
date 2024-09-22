import { app } from '../../../scripts/app.js'

function addResizeHook(node, padding) {
    let origOnCreated = node.onNodeCreated
    node.onNodeCreated = function() {
        let r = origOnCreated?.apply(this, arguments)
        let size = this.computeSize();
        size[0] += padding || 0;
        this.setSize(size);
        return r
    }
}


app.registerExtension({
    name: "AnimateDiffEvolved.autosize",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name?.startsWith("ADE_")) {
            if (nodeData?.input?.hidden?.autosize) {
                addResizeHook(nodeType.prototype, nodeData.input.hidden.autosize[1]?.padding)
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
