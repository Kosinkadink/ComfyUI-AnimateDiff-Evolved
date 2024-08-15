import { app } from '../../../scripts/app.js'
app.registerExtension({
    name: "AnimateDiffEvolved.autosize",
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
                let origOnCreated = node.onNodeCreated
                node.onNodeCreated = function() {
                    let r = origOnCreated?.apply(this, arguments)
                    let size = this.computeSize();
                    size[0] += inputData[1].padding || 0;
                    this.setSize(size);
                    return r
                }
                return w;
            }
        }
    }
});
