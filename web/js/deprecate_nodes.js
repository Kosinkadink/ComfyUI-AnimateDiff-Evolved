import { app } from '../../../scripts/app.js'

app.ui.settings.addSetting({
    id: "ADE.ShowExperimental",
    name: "🎭🅐🅓 Show experimental nodes",
    type: "boolean",
    defaultValue: false,
});
app.ui.settings.addSetting({
    id: "ADE.ShowDeprecated",
    name: "🎭🅐🅓 Show deprecated nodes",
    type: "boolean",
    defaultValue: false,
});

const deprecate_nodes = {
    name: 'AnimateDiff.deprecate_nodes',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        let showDeprecated = app.ui.settings.getSettingValue("ADE.ShowDeprecated", false)
        let showExperimental = app.ui.settings.getSettingValue("ADE.ShowExperimental", false)
        let warnWidget = false
        if (nodeData?.input?.optional) {
            for (let w in nodeData.input.optional) {
                if (nodeData.input.optional[w][0] == "ADEWARN") {
                    warnWidget = nodeData.input.optional[w]
                    break
                }
            }
        }
        if (warnWidget) {
            if (!((warnWidget[1].warn_type || "deprecated") == "deprecated" && showDeprecated) &&
                !(warnWidget[1].warn_type == "experimental" && showExperimental)) {
                nodeType.filter = "hidden"
            }
        }
        if (!nodeType.filter) {
            nodeType.filter = app.graph.filter
        }

    },
    async init() {
        app.graph.filter = app.graph.filter || "shown"
    },
    async setup() {
        for (let k in LiteGraph.registered_node_types) {
            let nodeType = LiteGraph.registered_node_types[k]
            if (!nodeType.filter) {
                nodeType.filter = app.graph.filter
            }
        }
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
                        var text_color = inputData[1]['color'] || "#FCC"
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
                            ctx.fillText(disp_text, widget_width/2, y + H * 0.7);
                            ctx.restore();
                        }

                    },
                    options : {"serialize": false},
                    computeSize : function(width) {
                        if (inputData[1]['text']) {
                            return [width, 20]
                        }
                        return [0, -4]

                    }
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

