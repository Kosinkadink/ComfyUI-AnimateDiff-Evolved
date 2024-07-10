import { app } from '../../../scripts/app.js'
app.registerExtension({
    name: "AnimateDiffEvolved.autosize",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData?.name?.startsWith("ADE_") ||
           nodeData?.name?.startsWith("ACN_")) {
            let origOnCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function() {
                const r = origOnCreated ? origOnCreated.apply(this) : undefined;
                this.setSize(this.computeSize(0));
                return r
            }
        }
    }
});
