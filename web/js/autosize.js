import { app } from '../../../scripts/app.js'
app.registerExtension({
    name: "AnimateDiffEvolved.autosize",
    async nodeCreated(node) {
        if(node.adeAutosize) {
            let size = node.computeSize(0);
            size[0] += node.adeAutosize?.padding || 0;
            node.setSize(size);
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
                node.adeAutosize = inputData[1];
                if (!node.widgets) {
                    node.widgets = []
                }
                node.widgets.push(w)
                return w;
            }
        }
    }
});
