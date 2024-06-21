import { app } from '../../../scripts/app.js'

const deprecate_nodes = {
    name: 'AnimateDiff.deprecate_nodes',
    async setup() {
        if (app.graph.filter) {
            //Someone else is doing a thing. Give up
            return
        }
        for (let k in LiteGraph.registered_node_types) {
            let n = LiteGraph.registered_node_types[k]
            let warnWidget = false
            if (n?.nodeData?.input?.optional) {
                for (let w in n.nodeData.input.optional) {
                    if (n.nodeData.input.optional[w][0] == "ADEWARN") {
                        warnWidget = true
                        break
                    }
                }
            }
            if (warnWidget) {
                n.filter = "hidden"
                continue
            }
            if (!n.filter) {
                n.filter = "shown"
            }
        }
        app.graph.filter = "shown"
    },
    async getCustomWidgets() {
        return {
            ADEWARN(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "ADE.WARN",
                    value : "",
                    draw : function(ctx, node, widget_width, y, H) {
                        var show_text = app.canvas.ds.scale > 0.5;
                        var margin = 15;
                        var text_color = "#FCC"
                        ctx.textAlign = "center";
                        if (show_text) {
                            if(!this.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();
                            ctx.fillStyle = text_color;
                            let disp_text = inputData[1]['text']
                            ctx.fillText(disp_text, widget_width/2, y + H * 0.7); //30 chars max
                            ctx.restore();
                        }

                    },
                    options : {}
                }
                if (!node.widgets) {
                    node.widgets = []
                }
                node.widgets.push(w)
                return w
            }
        }
    }
}

app.registerExtension(deprecate_nodes)

